"""Per-class PPE fusion with one vote per model and explicit disagreement."""
from __future__ import annotations
from ppe_classes import CLASS_IDS
from ppe_types import Detection, iou


def combine_ppe(primary, reference, vest_match_iou=.30, nms_iou=.45, additional=None):
    """Accept vest agreement from any two models; preserve single votes as UNKNOWN.

    Greedily match highest-IoU pairs across distinct models, then attach at most
    one third-model vote if it overlaps every member. Each detection votes once.
    Scores are the second-highest input score, not an inflated probability.
    SH17/reference alone supplies people; PPE from every model is retained.
    """
    sources=[primary,reference]
    if additional is not None:
        sources.append(additional)
    votes=[[d for d in src if d.name=='safety-vest'] for src in sources]
    edges=[]
    for a in range(len(votes)):
        for b in range(a+1,len(votes)):
            edges.extend((iou(x.xyxy,y.xyxy),a,i,b,j) for i,x in enumerate(votes[a])
                         for j,y in enumerate(votes[b]))
    used=set()
    vests=[]
    for overlap,a,i,b,j in sorted(edges,reverse=True):
        if overlap<vest_match_iou or (a,i) in used or (b,j) in used:
            continue
        cluster=[(a,i),(b,j)]
        used.update(cluster)
        for c in range(len(votes)):
            if c in (a,b):
                continue
            compatible=[(min(iou(d.xyxy,votes[s][k].xyxy) for s,k in cluster),n)
                        for n,d in enumerate(votes[c]) if (c,n) not in used]
            if compatible:
                similarity,n=max(compatible)
                if similarity>=vest_match_iou:
                    cluster.append((c,n))
                    used.add((c,n))
        matched=[votes[s][k] for s,k in cluster]
        box=tuple(sum(d.xyxy[k] for d in matched)/len(matched) for k in range(4))
        score=sorted((d.conf for d in matched),reverse=True)[1]
        vests.append(Detection(CLASS_IDS['safety-vest'],'safety-vest',score,box))
    for s,source in enumerate(votes):
        for k,d in enumerate(source):
            if (s,k) not in used:
                vests.append(Detection(CLASS_IDS['uncertain-vest'],'uncertain-vest',d.conf,d.xyxy))
    pooled=[]
    for s,source in enumerate(sources):
        pooled.extend(d for d in source if d.name!='safety-vest' and (d.name!='person' or s==1))
    kept=[]
    for d in sorted(pooled,key=lambda d:d.conf,reverse=True):
        if not any(k.name==d.name and iou(k.xyxy,d.xyxy)>nms_iou for k in kept):
            kept.append(d)
    return sorted(kept+vests,key=lambda d:d.conf,reverse=True)

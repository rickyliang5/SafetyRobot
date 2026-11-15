"""Conservative, per-class fusion of SH17 and SafetyVision detections.

Two-model vest agreement is an experimental precision/recall tradeoff, not a
probability estimate. Disagreement is explicit UNKNOWN evidence downstream.
"""
from __future__ import annotations
from ppe_classes import CLASS_IDS
from ppe_types import Detection, iou


def combine_ppe(primary, reference, vest_match_iou=.30, nms_iou=.45):
    """Use SH17 people/body parts, merge PPE, and require two-model vest agreement.

    Match vest boxes one-to-one, highest IoU first. Confidence is the weaker
    input score; agreeing models do not inflate it. Disagreeing boxes retain
    their location as uncertain-vest markers. They are not worn-PPE detections.
    """
    primary_vests = [d for d in primary if d.name == 'safety-vest']
    reference_vests = [d for d in reference if d.name == 'safety-vest']
    pairs = sorted(((iou(a.xyxy,b.xyxy),i,j) for i,a in enumerate(primary_vests)
                    for j,b in enumerate(reference_vests)), reverse=True)
    used_primary,used_reference = set(),set()
    vests=[]
    for overlap,i,j in pairs:
        if overlap < vest_match_iou or i in used_primary or j in used_reference:
            continue
        used_primary.add(i)
        used_reference.add(j)
        a,b=primary_vests[i],reference_vests[j]
        box=tuple((x+y)/2 for x,y in zip(a.xyxy,b.xyxy))
        vests.append(Detection(CLASS_IDS['safety-vest'],'safety-vest',min(a.conf,b.conf),box))
    for detections,used in ((primary_vests,used_primary),(reference_vests,used_reference)):
        for i,d in enumerate(detections):
            if i not in used:
                vests.append(Detection(CLASS_IDS['uncertain-vest'],'uncertain-vest',d.conf,d.xyxy))
    # Original body-part evidence remains available; candidate explicit NO-*
    # classes keep their own names and evidence thresholds. People come only
    # from SH17, avoiding the candidate's weak Person class.
    pooled=[d for d in primary if d.name not in ('person','safety-vest')]
    pooled += [d for d in reference if d.name != 'safety-vest']
    kept=[]
    for d in sorted(pooled,key=lambda d:d.conf,reverse=True):
        if not any(k.name==d.name and iou(k.xyxy,d.xyxy)>nms_iou for k in kept):
            kept.append(d)
    return sorted(kept+vests,key=lambda d:d.conf,reverse=True)

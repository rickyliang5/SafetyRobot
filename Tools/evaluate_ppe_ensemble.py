"""Visual evaluation of existing PPE models, their ensemble and two candidates.

Unlabeled footage: counts are not recall/precision. Candidate weights stay local.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from compare_ppe_models import Detector, draw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Code/Server'))
from ppe_config import load_config
from ppe_detector import PpeDetector
from ppe_ensemble import combine_ppe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--stride',type=int,default=10)
    ap.add_argument('--out',type=Path,default=ROOT/'ppe_results/ensemble_evaluation')
    args=ap.parse_args()
    if args.stride<1:
        ap.error('stride must be positive')
    cfg=load_config().model
    models={
        'SH17':PpeDetector(ROOT/'Code/Server/models/ppe_yolo8n_640.onnx',conf=cfg.conf_default,class_conf=cfg.class_conf),
        'SafetyVision':PpeDetector(ROOT/'Code/Server/models/ppe_safetyvision_nano_640.onnx',conf=cfg.conf_default,class_conf=cfg.class_conf),
        'Kaggle construction nano':Detector(ROOT/'test_models/construction_nano/best.onnx',preprocessing='stretch'),
        'PPE YOLO11s':Detector(ROOT/'test_models/ppe_yolo11s/best.onnx'),
    }
    names=['SH17','SafetyVision','Ensemble','Kaggle construction nano','PPE YOLO11s']
    for model in list(models.values())[:2]:
        model.warmup(5)
    cap=cv2.VideoCapture(str(args.source))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps<=0:
        raise RuntimeError('cannot read source video/frame rate')
    args.out.mkdir(parents=True,exist_ok=True)
    counts={n:Counter() for n in names}
    vest_frames=Counter()
    timings={n:[] for n in names}
    writer=None
    index=processed=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            source_index=index
            index+=1
            if source_index%args.stride:
                continue
            detections={}
            for name,model in list(models.items())[:2]:
                start=time.perf_counter()
                detections[name]=model.detect(frame)
                timings[name].append((time.perf_counter()-start)*1000)
            start=time.perf_counter()
            detections['Ensemble']=combine_ppe(detections['SafetyVision'],detections['SH17'],cfg.vest_match_iou,cfg.iou_nms)
            timings['Ensemble'].append(timings['SH17'][-1]+timings['SafetyVision'][-1]+(time.perf_counter()-start)*1000)
            display={}
            for name in names[:3]:
                display[name]=[('vest' if d.name=='safety-vest' else 'vest?' if d.name=='uncertain-vest' else d.name,d.conf,d.xyxy)
                               for d in detections[name] if d.name in ('helmet','safety-vest','uncertain-vest')]
            for name,model in list(models.items())[2:]:
                start=time.perf_counter()
                raw=model.detect(frame,cfg.conf_default,cfg.iou_nms)
                display[name]=[d for d in raw if d[1]>=cfg.conf_for('safety-vest' if d[0]=='vest' else 'helmet')]
                timings[name].append((time.perf_counter()-start)*1000)
                model.timings.clear()
            panes=[]
            for name in names:
                counts[name].update(d[0] for d in display[name])
                vest_frames[name]+=int(any(d[0]=='vest' for d in display[name]))
                panes.append(draw(frame,display[name],f'{name} | {source_index/fps:.1f}s'))
            note=np.zeros_like(panes[0])
            for y,text in enumerate(['Worn PPE only; vest? = disagreement',
                                     f'Vest threshold {cfg.conf_for("safety-vest"):.2f}; helmet {cfg.conf_for("helmet"):.2f}',
                                     'Ensemble requires two-model vest IoU >= 0.30',
                                     'Counts are not measured accuracy',
                                     'No labels or Pi performance measurements']):
                cv2.putText(note,text,(12,40+y*35),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            panes.append(note)
            canvas=np.vstack((np.hstack(panes[:3]),np.hstack(panes[3:])))
            if writer is None:
                writer=cv2.VideoWriter(str(args.out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.stride,(canvas.shape[1],canvas.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('cannot write comparison.mp4')
            writer.write(canvas)
            if processed%15==0:
                if not cv2.imwrite(str(args.out/f'frame_{source_index:06d}.jpg'),canvas):
                    raise RuntimeError('cannot save sample frame')
            processed+=1
            if processed%30==0:
                print(f'{processed} frames processed, source {source_index/fps:.1f}s',flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for m in list(models.values())[:2]:
            m.close()
    if not processed:
        raise RuntimeError('no frames processed')
    report={'source':str(args.source.resolve()),'sampled_frames':processed,'stride':args.stride,
            'thresholds':{'global':cfg.conf_default,'class_conf':cfg.class_conf,'vest_match_iou':cfg.vest_match_iou},
            'detections':{k:dict(v) for k,v in counts.items()},'frames_with_vest':dict(vest_frames),
            'timing_ms':{n:{'mean':round(float(np.mean(t)),2),'p95':round(float(np.percentile(t,95)),2)} for n,t in timings.items()},
            'candidate_hashes':{n:m.sha256 for n,m in list(models.items())[2:]},
            'limitations':'Unlabeled footage. No claims of precision/recall or shirt-vs-vest correctness. Laptop CPU timing only; ensemble time includes both inference passes. Construction nano uses author-specified stretch preprocessing.'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

"""Compare SH17 with a metadata-labelled YOLOv8 ONNX PPE candidate.

Visual evaluation only: counts are per-frame boxes, not accuracy or people.
Uses shared preprocessing, class-wise NMS and identical confidence thresholds.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code/Server'))
from ppe_detector import preprocess, unletterbox_boxes, nms_numpy

ALIASES = {'helmet': 'helmet', 'hardhat': 'helmet',
           'safety-vest': 'vest', 'safety vest': 'vest', 'vest': 'vest'}


class Detector:
    def __init__(self, path, threads=3):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 640, 640] or inp.type != 'tensor(float)':
            raise ValueError(f'expected float32 1x3x640x640, got {inp.shape} {inp.type}')
        self.input_name = inp.name
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta['names'])
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names = {int(k): v for k,v in names.items()}
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for _ in range(5):
            self.session.run(None, {self.input_name: np.zeros((1,3,640,640), np.float32)})
        self.timings = []

    def detect(self, frame, conf, iou):
        start = time.perf_counter()
        tensor, ratio, pad = preprocess(frame, (640,640))
        raw = self.session.run(None, {self.input_name: tensor})[0]
        arr = raw[0]
        nc = len(self.names)
        if arr.shape[0] == 4+nc:
            arr = arr.T
        if arr.shape[1] != 4+nc:
            raise ValueError(f'unsupported raw YOLO output {raw.shape}')
        ids = arr[:,4:].argmax(axis=1)
        scores = arr[np.arange(len(arr)), ids+4]
        selected = scores >= conf
        coords, ids, scores = arr[selected,:4], ids[selected], scores[selected]
        xyxy = np.concatenate((coords[:,:2]-coords[:,2:]/2, coords[:,:2]+coords[:,2:]/2),axis=1)
        xyxy = unletterbox_boxes(xyxy, ratio, pad, (frame.shape[1],frame.shape[0]))
        dets = []
        for cid in np.unique(ids):
            label = ALIASES.get(self.names[int(cid)].lower())
            if label is None:
                continue
            sel = np.where(ids == cid)[0]
            for k in nms_numpy(xyxy[sel], scores[sel], iou):
                j = sel[k]
                box = xyxy[j]
                if box[2]-box[0] >= 1 and box[3]-box[1] >= 1:
                    dets.append((label, float(scores[j]), box.tolist()))
        self.timings.append((time.perf_counter()-start)*1000)
        return dets


def draw(frame, dets, title):
    width = 640
    scale = width/frame.shape[1]
    pane = cv2.resize(frame, (width, round(frame.shape[0]*scale)))
    for label, conf, box in dets:
        x1,y1,x2,y2 = [round(v*scale) for v in box]
        color = (50,220,70) if label == 'vest' else (240,190,30)
        cv2.rectangle(pane,(x1,y1),(x2,y2),color,1)
        text = f'{label} {conf:.2f}'
        cv2.putText(pane,text,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3)
        cv2.putText(pane,text,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
    banner = np.zeros((48,width,3),np.uint8)
    cv2.putText(banner,title,(8,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return np.vstack((banner,pane))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, default=ROOT/'Code/Server/models/ppe_yolo8n_640.onnx')
    ap.add_argument('--out', type=Path, default=ROOT/'ppe_results/model_comparison')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--conf', type=float, default=.25)
    ap.add_argument('--iou', type=float, default=.45)
    args = ap.parse_args()
    if args.stride < 1 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        ap.error('invalid stride/conf/iou')
    models = {'SH17 nano': Detector(args.baseline), 'SafetyVision nano': Detector(args.candidate)}
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        ap.error('cannot read source video/frame rate')
    args.out.mkdir(parents=True,exist_ok=True)
    counts = {name:Counter() for name in models}
    vest_frames = Counter()
    samples = []
    index = processed = 0
    writer = None
    try:
        while True:
            ok,frame = cap.read()
            if not ok:
                break
            source_index = index
            index += 1
            if source_index % args.stride:
                continue
            panes = []
            found = {}
            for name,model in models.items():
                dets = model.detect(frame,args.conf,args.iou)
                found[name] = sum(d[0]=='vest' for d in dets)
                counts[name].update(d[0] for d in dets)
                vest_frames[name] += int(found[name]>0)
                panes.append(draw(frame,dets,f'{name} | conf {args.conf:.2f} | {source_index/fps:.1f}s'))
            comparison = np.hstack(panes)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.stride,(comparison.shape[1],comparison.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('cannot write comparison.mp4')
            writer.write(comparison)
            if processed%30 == 0:
                p = args.out/f'frame_{source_index:06d}.jpg'
                if not cv2.imwrite(str(p),comparison):
                    raise RuntimeError(f'cannot write {p}')
                samples.append(str(p))
            processed += 1
            if processed%100 == 0:
                print(f'{processed} sampled frames processed',flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if processed == 0:
        raise RuntimeError('no frames processed')
    report = {'source':str(args.source.resolve()),'sampled_frames':processed,
              'stride':args.stride,'conf':args.conf,'iou':args.iou,
              'runtime':'ONNX Runtime CPU, 3 intra-op threads, warmup=5, 640 input',
              'counts':{k:dict(v) for k,v in counts.items()},'frames_with_vest':dict(vest_frames),
              'models':{k:{'names':m.names,'onnx_sha256':m.sha256,
                           'mean_ms':float(np.mean(m.timings)), 'p95_ms':float(np.percentile(m.timings,95))}
                        for k,m in models.items()}, 'samples':samples,
              'limitations':'Unlabeled visual comparison, not accuracy. Laptop CPU timings are not Pi performance. PPE-worn boxes only; no violation classes, tracking, or compliance rules.'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('models','samples')},indent=2))
    for k,m in models.items():
        print(f'{k}: mean {np.mean(m.timings):.1f}ms; p95 {np.percentile(m.timings,95):.1f}ms')


if __name__ == '__main__':
    main()

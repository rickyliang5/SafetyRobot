"""Compare previous and configured PPE thresholds on a local video.

Uses the original PyTorch checkpoint on PC; does not measure Pi/ONNX speed
or accuracy. Counts are detections across sampled frames, not unique people.
Example: python Tools/test_ppe_video.py --source test_videos/construction.mp4
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code' / 'Server'))
from ppe_config import load_config
from ppe_classes import CLASS_NAMES
from ppe_detector import annotate
from ppe_types import Detection

PPE = {'ear-mufs', 'face-guard', 'face-mask', 'glasses', 'gloves',
       'helmet', 'medical-suit', 'shoes', 'safety-suit', 'safety-vest'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--stride', type=int, default=5, help='process every Nth frame')
    ap.add_argument('--max-frames', type=int, default=0, help='0 means all sampled frames')
    ap.add_argument('--out', type=Path, default=ROOT / 'ppe_results/video_thresholds')
    args = ap.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        ap.error('stride must be positive and max-frames nonnegative')
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        ap.error(f'cannot open video: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        ap.error('source has no usable frame rate')
    cfg = load_config(args.config).model
    baseline = replace(cfg, conf_default=0.30,
                       class_conf={**cfg.class_conf, 'safety-vest': 0.40},
                       veto_conf={**cfg.veto_conf, 'safety-vest': 0.35})
    from ultralytics import YOLO
    model = YOLO(str(ROOT / 'yolo8n.pt'))
    if tuple(model.names[i] for i in range(len(model.names))) != CLASS_NAMES:
        raise ValueError('checkpoint class order differs from the robot detector')
    args.out.mkdir(parents=True, exist_ok=True)
    writer = None
    counts = {'before': Counter(), 'after': Counter()}
    vest_frames = Counter()
    sample_paths = []
    index = processed = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_index = index
            index += 1
            if source_index % args.stride:
                continue
            result = model.predict(frame, imgsz=640,
                                   conf=min(baseline.conf_default, cfg.conf_default),
                                   iou=cfg.iou_nms, verbose=False, device='cpu')[0]
            all_dets = [Detection(cls_id=int(b.cls.item()),
                                  name=model.names[int(b.cls.item())],
                                  conf=float(b.conf.item()),
                                  xyxy=tuple(float(v) for v in b.xyxy[0].tolist()))
                        for b in result.boxes]
            panes = []
            kept = {}
            for label, settings in [('before', baseline), ('after', cfg)]:
                dets = [d for d in all_dets if d.name in PPE and
                        d.conf >= max(settings.conf_default, settings.conf_for(d.name))]
                kept[label] = dets
                counts[label].update(d.name for d in dets)
                vest_frames[label] += int(any(d.name == 'safety-vest' for d in dets))
                pane = annotate(frame, dets)
                pane = cv2.resize(pane, (640, max(1, round(frame.shape[0]*640/frame.shape[1]))))
                banner = np.zeros((48, 640, 3), dtype=np.uint8)
                title = f'{label.upper()} | vest >= {max(settings.conf_default, settings.conf_for("safety-vest")):.2f} | {source_index/fps:.1f}s'
                cv2.putText(banner, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
                panes.append(np.vstack((banner, pane)))
            comparison = np.hstack(panes)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / 'comparison.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps/args.stride,
                                         (comparison.shape[1], comparison.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('cannot create comparison.mp4')
            writer.write(comparison)
            new_vest = sum(d.name == 'safety-vest' for d in kept['after']) > sum(d.name == 'safety-vest' for d in kept['before'])
            if processed % 60 == 0 or (new_vest and len(sample_paths) < 20):
                path = args.out / f'frame_{source_index:06d}.jpg'
                if not cv2.imwrite(str(path), comparison):
                    raise RuntimeError(f'cannot write {path}')
                sample_paths.append(str(path))
            processed += 1
            if processed % 50 == 0:
                print(f'processed {processed} sampled frames ({source_index/fps:.1f}s of source)', flush=True)
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not processed:
        raise RuntimeError('no readable video frames')
    report = {'source': str(args.source.resolve()), 'sampled_frames': processed,
              'stride': args.stride, 'source_fps': fps,
              'elapsed_s': round(time.perf_counter()-started, 2),
              'thresholds': {label: {'global': s.conf_default, 'class_conf': s.class_conf}
                             for label,s in [('before',baseline),('after',cfg)]},
              'detections': {k: dict(v) for k,v in counts.items()},
              'frames_with_vest': dict(vest_frames), 'sample_images': sample_paths,
              'limitations': 'Unlabeled visual comparison. More boxes do not prove better accuracy. No compliance decisions, tracking, or Pi performance evaluation.'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('thresholds','sample_images')}, indent=2))
    print(f'Results: {args.out.resolve()}')


if __name__ == '__main__':
    main()

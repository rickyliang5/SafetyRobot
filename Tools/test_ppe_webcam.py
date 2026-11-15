"""Interactive laptop webcam preview using the SafetyVision ONNX model.

Run from the repo root: .venv-ppe/Scripts/python Tools/test_ppe_webcam.py
Q/Esc closes the camera. Confidence slider controls visible detections.
No video or snapshots are saved. --video and --headless support smoke testing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import compare_ppe_models as comparison

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'SafetyRobot - Laptop PPE Test (Q or Esc to stop)'
# Only worn PPE is shown. Person/body-part and missing-PPE classes are excluded.
comparison.ALIASES.update({'gloves':'gloves', 'goggles':'goggles', 'mask':'mask'})
COLORS = {'helmet':(240,190,30), 'vest':(50,220,70),
          'gloves':(220,100,220), 'goggles':(70,210,240), 'mask':(200,180,70)}


def open_capture(camera, video):
    if video:
        cap = cv2.VideoCapture(str(video))
        ok,frame = cap.read()
        if ok:
            return cap,frame
        cap.release()
        raise RuntimeError(f'cannot read video: {video}')
    # Windows laptop cameras vary in which capture backend they support.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(camera, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            for _ in range(15):
                ok,frame = cap.read()
                if ok:
                    return cap,frame
        cap.release()
    raise RuntimeError(f'cannot read camera {camera}; close other camera apps and check Windows camera permissions')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera',type=int,default=0)
    ap.add_argument('--model',type=Path,default=ROOT/'Code/Server/models/ppe_safetyvision_nano_640.onnx')
    ap.add_argument('--conf',type=float,default=.25)
    ap.add_argument('--video',type=Path,help='use a local video instead of the camera')
    ap.add_argument('--headless',action='store_true',help='do not open a preview window')
    ap.add_argument('--max-frames',type=int,default=0,help='0 runs until closed')
    args = ap.parse_args()
    if not 0 < args.conf <= 1 or args.max_frames < 0:
        ap.error('confidence must be (0,1]; max-frames must be nonnegative')
    if args.headless and not args.video and args.max_frames == 0:
        ap.error('headless camera testing requires --max-frames')
    print(f'Loading model: {args.model}',flush=True)
    model = comparison.Detector(args.model)
    cap = None
    count = 0
    last_tick = time.perf_counter()
    try:
        cap,frame = open_capture(args.camera,args.video)
        if not args.headless:
            cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE,1100,680)
            cv2.createTrackbar('Confidence %',TITLE,round(args.conf*100),100,lambda _:None)
        while True:
            conf = args.conf if args.headless else max(.01,cv2.getTrackbarPos('Confidence %',TITLE)/100)
            dets = model.detect(frame,conf,.45)
            infer_ms = model.timings[-1]
            model.timings.clear()  # Keep memory bounded during long camera sessions.
            count += 1
            if count == 1:
                print(f'READY: {frame.shape[1]}x{frame.shape[0]}; camera={args.camera if not args.video else args.video}; worn PPE only; no recording',flush=True)
            if not args.headless:
                view = frame.copy()
                for label,score,box in dets:
                    x1,y1,x2,y2 = [round(v) for v in box]
                    color = COLORS.get(label,(80,220,110))
                    cv2.rectangle(view,(x1,y1),(x2,y2),color,2)
                    text = f'{label} {score:.0%}'
                    pos = (x1,max(20,y1-6))
                    cv2.putText(view,text,pos,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
                    cv2.putText(view,text,pos,cv2.FONT_HERSHEY_SIMPLEX,.6,color,1)
                now = time.perf_counter()
                fps = 1/max(now-last_tick,1e-6)
                last_tick = now
                banner = f'SafetyVision nano | threshold {conf:.0%} | {fps:.1f} FPS | {infer_ms:.0f} ms | Q/Esc: stop'
                cv2.rectangle(view,(0,0),(view.shape[1],34),(25,25,25),-1)
                cv2.putText(view,banner,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                cv2.imshow(TITLE,view)
                key = cv2.waitKey(1)&0xff
                if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(TITLE,cv2.WND_PROP_VISIBLE)<1:
                    break
            if args.max_frames and count>=args.max_frames:
                break
            ok,frame = cap.read()
            if not ok:
                if args.video:
                    break
                raise RuntimeError('camera stopped delivering frames')
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'Stopped after {count} frames; camera released.',flush=True)


if __name__ == '__main__':
    main()

"""Live model preview on the Pi camera, without motors or recording."""
from __future__ import annotations

import argparse
import signal
import time

import cv2

from ppe_camera import open_camera
from ppe_config import load_config
from ppe_detector import PpeDetector, annotate
from ppe_rules import RuleEngine
from ppe_tracker import PersonTracker, ViolationConfirmer
from ppe_mjpeg import FrameBuffer, MjpegServer, encode_jpeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--source', choices=('picamera', 'webcam'), default='picamera')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--duration', type=float, default=0, help='seconds; 0 runs until stopped')
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_label = "PPE ensemble (SH17 + SafetyVision)" if cfg.model.ensemble_ppe else "SafetyVision nano"
    detector = PpeDetector.from_config(cfg)
    detector.warmup()
    rules = RuleEngine(cfg.rules, cfg.model)
    tracker = PersonTracker(window=cfg.temporal.window_m,
                            presence_hold_s=cfg.temporal.presence_hold_s)
    confirmer = ViolationConfirmer(cfg.temporal, cfg.cooldown)
    camera = open_camera(cfg, args.source)
    buffer = FrameBuffer()
    server = MjpegServer(buffer, port=args.port)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    frames = 0
    try:
        camera.start()
        server.start()
        while running and (not args.duration or time.monotonic() - started < args.duration):
            frame = camera.capture_bgr()
            if frame is None:
                raise RuntimeError('Camera stopped delivering frames')
            tick = time.perf_counter()
            detections = detector.detect(frame)
            elapsed = time.perf_counter() - tick
            now = time.monotonic()
            observations = rules.evaluate_frame(detections, (frame.shape[1], frame.shape[0]))
            pairs = tracker.update(observations, now)
            view = annotate(frame, detections)
            for track, obs in pairs:
                missing = confirmer.confirmed_items(track)
                labels = []
                for item in obs.items:
                    stamp = track.last_present.get(item, float('-inf'))
                    if obs.items[item].status == 'PRESENT' or now - stamp < cfg.temporal.presence_hold_s:
                        state = 'OK' if obs.items[item].status == 'PRESENT' else 'OK (recent)'
                    elif item in missing:
                        state = 'MISSING'
                    else:
                        state = 'CHECKING'
                    labels.append(f'{item}: {state}')
                x, y = int(obs.person.x1), max(45, int(obs.person.y1))
                for row, label in enumerate([f'Person {track.track_id}'] + labels):
                    pos = (x, y + row * 20)
                    cv2.putText(view, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(view, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (80, 240, 140), 1, cv2.LINE_AA)
            status = f'{model_label} | {elapsed * 1000:.0f} ms | camera preview'
            cv2.putText(view, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255, 255, 255), 1, cv2.LINE_AA)
            jpeg = encode_jpeg(view)
            if jpeg is None:
                raise RuntimeError('Could not encode camera frame')
            buffer.status_line = status
            buffer.publish(jpeg)
            frames += 1
            if frames == 1:
                print(f'READY: live camera on port {args.port}; model={model_label}', flush=True)
    finally:
        server.stop()
        camera.close()
        print(f'Camera released after {frames} frames.', flush=True)


if __name__ == '__main__':
    main()

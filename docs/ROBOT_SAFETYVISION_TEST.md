# SafetyVision nano robot test branch

Branch: robot/safetyvision-nano

This branch loads SafetyVision v1 YOLOv8-nano for PPE and explicit missing-PPE
predictions. It retains SH17 nano for person detection, because the candidate
missed all people in our group-photo smoke test despite finding their PPE.
Both self-contained ONNX files and their provenance metadata are committed
under Code/Server/models. No checkpoint download or PyTorch install is needed
on the Pi. Two inference passes cost more than the earlier single-model timing.

The default branch configuration requires helmet and safety-vest, with global
confidence 0.25 and vest confidence/presence threshold 0.25. Helmet detection
stays at 0.40 and person detection at 0.45. Explicit NO-* labels need 0.65,
clear person visibility and unambiguous association to count as missing-PPE
evidence. Worn PPE presence wins over conflicting explicit NO-* predictions.
The old bare-body geometry remains available for SH17 models. Explicit model
classifications are not converted into bare head/torso boxes.

Temporal confirmation is unchanged: helmet 5/10 frames; vest 8/10, with
recording only while stopped/settled. Unknown/ambiguous views remain unknown.
Fall and harness classes are deliberately ignored. All scores remain model
predictions, not guarantees of compliance or calibrated probabilities.

## On the Pi

For a camera-only browser preview (no motors, servos, uploads or saved images):

```bash
cd ~/SafetyRobot-safetyvision/Code/Server
~/ppe-venv/bin/python ppe_preview.py --config ppe_config.json
```

Open http://PI_IP:8080/ in a browser. The preview draws live model predictions
and inference speed. Ctrl-C releases the camera. Use `--duration 15` for a
short capture check. The camera flips match this robot's mounted orientation.

From your existing SafetyRobot checkout:

```bash
git fetch origin
git switch robot/safetyvision-nano
git pull --ff-only
~/ppe-venv/bin/pip install --no-deps 'onnxruntime>=1.18,<2'
cd Code/Server
~/ppe-venv/bin/python ppe_detector.py --bench --runs 50
~/ppe-venv/bin/python ppe_main.py --config ppe_config.json --no-motors --no-upload -v
```

The explicit --config uses the checked-in test settings instead of an old
ppe_config.local.json that may still point to SH17-only paths. On this first
run, wheels stay disabled and snapshots remain local. Live annotated view:
http://PI_IP:8080/stream.mjpg . Ctrl-C stops capture and releases hardware.
For your normal patrol run, retain any needed local portal/robot settings but
set model.confirm_onnx and model.scan_onnx to models/ppe_safetyvision_nano_640.onnx,
and model.person_onnx to models/ppe_yolo8n_640.onnx. Choose required_ppe explicitly.

The candidate uses ONNX IR 10, so runtime 1.17 is insufficient. The --no-deps
upgrade preserves the Pi's system NumPy/OpenCV camera stack. No torch or
ultralytics is needed on the Pi. Benchmarks include both detector passes;
watch actual Pi latency/temperature before selecting a sustained frame rate.

## Tests and limitations

The laptop smoke test found five person owners in the group photo using SH17,
with candidate PPE attached through the existing rule engine. Simulated patrol
with saved footage started, detected a person, entered HOLD, and shut down cleanly.
A thread event named _stop previously collided with threading.Thread._stop;
drive, vision and uploader now use _stop_event.

Regression tests cover class remapping, supported vocabularies, unsupported
labels, canonical allowed-class filtering, explicit violation semantics,
positive-presence conflicts, confidence/view gates, separate person-model
selection, and actual drive-thread start/stop. Laptop results do not establish
Pi performance or field accuracy. The candidate can mistake bright shirts for
vests and misses PPE in dust. See PPE_MODEL_COMPARISON.md for the earlier
single-model visual evaluation (its latency excludes the added person model).

The model is SafetyVision v1 (nano), not v2 (small):
https://huggingface.co/ayushgupta7777/safetyvision-yolov8
License in embedded model metadata: AGPL-3.0. Source URLs and file hashes are
in each .meta.json file. Downloads/outputs/webcam recordings are not committed.

To return to the previous robot version, stop the process and switch back to
main. Your gitignored local configuration and snapshots remain on the Pi.

## Camera preview with PPE memory

Run `~/ppe-venv/bin/python ppe_preview.py --config ppe_config.json` from
Code/Server and open http://PI_IP:8080. This runs capture and inference without
motors, uploads or recording. Per-person item labels hold OK for
`temporal.presence_hold_s` seconds (default 4) after a real positive observation.
Held labels say OK (recent); only real sightings refresh the timer. UNKNOWN
and missing detections do not refresh it. After expiry the label becomes
CHECKING until missing evidence meets the usual confirmation threshold.
Memory is separate for each tracked person and item, and clears after a long
observation gap. Raw detection boxes can still flicker; labels show the stable
assessment. Tracker identity mistakes and incorrect model detections can still
produce incorrect assessments. This is short-term memory, not a permanent
assumption that PPE remains worn. The patrol uses the same hold to suppress
missing-PPE reports, without changing raw observations or evidence counts.

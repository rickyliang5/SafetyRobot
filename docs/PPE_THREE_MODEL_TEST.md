# Three-model PPE robot test

Branch: robot/ppe-ensemble-construction
Parent: robot/ppe-ensemble. Main and both earlier robot-test branches are unchanged.

This branch includes and runs all three nano ONNX models:

1. SH17: person owners, body parts, and PPE predictions.
2. SafetyVision v1: PPE and explicit missing-PPE predictions.
3. Kaggle construction nano: the bottom-left candidate in our earlier comparison.
   Source: https://huggingface.co/mancalazure/construction-ppe-safety

The new model is committed as models/ppe_construction_nano_640.onnx, with source,
license information and SHA-256 in the adjacent .meta.json. It is the stock raw
YOLO output export, not the incompatible DeepStream graph. All three ONNX files
are self-contained; no PyTorch installation or extra model downloads are needed.

## Fusion behavior

- Any two of the three models can confirm a vest, with matched boxes at IoU>=0.30.
- Each detection is used once and each model contributes at most one vote per
  matched vest. A third match must overlap every existing cluster member.
- A vest found by only one model is shown as uncertain-vest (vest? in the laptop
  preview) and yields UNKNOWN rather than being treated as worn or missing.
- Two votes use the weaker score; three votes use the second-highest score.
  Coordinates are averaged. These are not calibrated probability estimates.
- Helmet and other PPE detections from all models are merged with class-wise
  NMS. Their selected original scores are retained. SH17 remains the sole source
  of person owners. Explicit no-PPE labels remain separate evidence classes.
- Existing ambiguity, visibility, positive-presence veto and temporal gates remain
  in force. A disagreement does not override an already agreed worn vest.

Construction nano requires a stretched 640x640 RGB input. SH17/SafetyVision use
letterboxing. The robot adapter selects the correct preprocessing from the exact
known class vocabulary and maps construction boxes back with separate x/y scales.
Unsupported machinery, vehicles and cones are ignored.

The active configuration adds model.construction_onnx and keeps ensemble_ppe=true.
Set construction_onnx to null to restore the two-model ensemble. Disabling
ensemble_ppe while retaining a construction model is rejected as contradictory.

## Pi setup

From the existing SafetyRobot folder, stop any running patrol first:

```bash
git fetch origin
git switch robot/ppe-ensemble-construction
git pull --ff-only
~/ppe-venv/bin/pip install --no-deps 'onnxruntime>=1.18,<2'
cd Code/Server
~/ppe-venv/bin/python ppe_detector.py --bench --runs 50
~/ppe-venv/bin/python ppe_main.py --config ppe_config.json --no-motors --no-upload -v
```

The explicit config loads this branch's three-model settings instead of stale
local model overrides. The initial test keeps wheels disabled and uploads off.
View the annotated feed at http://PI_IP:8080/stream.mjpg . Ctrl-C stops the process.
The config requires helmets and safety vests. Existing confirmation is helmet
5/10, vest 8/10 stopped/settled frames. For normal patrol, retain your local robot
and portal settings while copying the branch's model settings explicitly.

This adds a third inference pass and another roughly 12 MB ONNX file. Runtime
memory exceeds file size. A laptop simulated frame took roughly 127 ms in one
observed tick; that is not a Pi FPS benchmark or an accuracy measurement. Measure
latency and sustained temperature on your own Pi before choosing a patrol rate.
The --no-deps runtime install preserves the system NumPy/OpenCV camera stack.

Laptop preview:

```powershell
.\.venv-ppe\Scripts\python.exe Tools\test_ppe_webcam.py --ensemble
```

The preview loads all three models from the checked-in configuration. The slider
sets display candidate thresholds for all three, separately from robot per-class
thresholds. Only worn PPE and vest? markers are shown. Q/Esc closes the camera;
no video is recorded.

## Validation and limitations

131 tests passed: decoding, stretch geometry/RGB handling, exact class mapping,
two-of-three/one-vote-per-model fusion, unchanged rule/tracker/patrol behavior,
and thread shutdown. A full simulated patrol using saved construction footage
loaded all three models, detected a person, entered HOLD and stopped cleanly.
A five-frame preview smoke test passed. The construction adapter's worn-PPE
boxes/scores matched the earlier research implementation on frames 150, 600 and
1500. The ONNX graph checker passed (IR 7, opset 12, no external weight data).

This experiment is not a guarantee of higher accuracy. All three can share
clothing-label errors; a consensus can still mistake bright clothing for a vest.
The existing safety-suit/medical-suit acceptance policy is unchanged. The dusty
scene remains challenging. Additional votes can recover some two-model
disagreements, but weak/distant vests can still be UNKNOWN. Label Pi-camera
frames to measure actual vest precision/recall before declaring a winner.

The earlier five-way video and PPE_ENSEMBLE.md document the previous TWO-model
experiment and individual candidates. They are not a benchmark of this active
three-model pipeline. This branch adds construction nano, not the larger YOLO11s
candidate that missed vests in our checks.

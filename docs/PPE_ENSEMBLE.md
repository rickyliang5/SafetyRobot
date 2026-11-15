# PPE ensemble experiment and model research

Branch: robot/ppe-ensemble, based on robot/safetyvision-nano. Main and the
previous robot-test branch are preserved. This is an experiment, not a measured
accuracy upgrade. It combines predictions; it does not merge neural weights.

## What is combined

Both existing ONNX models already ran on each robot frame. Previously SH17
provided only people and SafetyVision provided PPE. With model.ensemble_ppe=true:

- SH17 supplies people, bare body parts, shoes and its other PPE detections.
- SafetyVision supplies PPE and separate explicit missing-PPE classes.
- Same-class helmet/other PPE boxes are merged with class-wise NMS, preserving
  the selected original model score. Confidence is not multiplied or increased.
- A vest must have a one-to-one match between the two models at IoU >=0.30.
  Agreed boxes use mean coordinates and the weaker score (no inflated confidence).
- Single-model vest votes are uncertain-vest markers, shown as vest? in the
  comparison and webcam preview. They count as neither worn nor missing PPE.
  The rule engine returns UNKNOWN for disagreements unless an agreed worn vest
  already clears that person. Explicit no-vest does not override disagreement.
- If neither model finds a vest, the existing clear-view/8-of-10 absence logic
  remains in effect. Disagreement is not a guarantee against every false alert.

This reuses the existing two inference passes rather than adding a third model
on the Pi. Matching can reduce accepted detections of real vests. It cannot
remove a mistake both models make, or establish that each selected prediction
is the best of the two. Existing safety-suit/medical-suit policy is unchanged.

Configuration: ensemble_ppe=false restores the previous person+PPE arrangement.
vest_match_iou is configurable; no value here is tuned on labeled footage.

## Other projects examined

1. Kaggle YOLOv8 PPE notebook using Construction Site Safety data:
   https://www.kaggle.com/code/daffamuhamadazhar/yolov8-ppe-detection
   Search results expose training logs; the page body was not readable by the
   browser tool. No verified vest-only test result was extracted from it.
2. Additional Kaggle-trained construction YOLOv8n ONNX export:
   https://huggingface.co/mancalazure/construction-ppe-safety
   Embedded metadata confirms /kaggle/working/ppe_data.yaml. The author publishes
   no held-out validation metrics and documents limitations. This model requires
   stretch preprocessing rather than our normal letterbox path, which the
   evaluation script supports explicitly. Only best.onnx is used, never the
   incompatible DeepStream export. It still labels high-visibility shirts vests.
3. PPE YOLO11s, approximately 38 MB ONNX:
   https://huggingface.co/nduka1999/nd_ppe_yolo11s
   The author reports 93.2% overall test mAP50 and >91% vest/no-vest AP50.
   These are author claims, not reproduced results. With the file's own metadata
   and 640 input, it produced zero vest detections in our 210 sampled frames and
   zero on all three clear vest photos. On the group photo its maximum raw
   vest-channel score was about 0.000035. Reject for deployment pending provenance,
   class-order/export verification; do not reinterpret channels to force boxes.
4. Six-class YOLOv8n card reports about 90% vest AP but also acknowledges lighting
   and motion limitations:
   https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes
   The live file list had only a .pt checkpoint, despite discussing ONNX. It was
   not loaded or evaluated; do not describe its headline score as reproduced.

A label-policy issue: Ultralytics' current Construction-PPE description explicitly
includes high-visibility jackets/shirts under vest. This does not prove that all
models share that exact definition, but means a vest-only requirement may differ
from training labels. No source establishes that a model is immune to shirt/vest
confusion on our camera.
https://www.ultralytics.com/blog/missing-ppe-detection-yolo

## Same-footage comparison

210 frames, stride 10, across the roughly 70-second construction career clip:
https://www.youtube.com/watch?v=s1n6nBhVzdo
Global/vest score 0.25; helmet score 0.40. Each model uses its own metadata and
supported preprocessing. Counts below are boxes across frames, not people,
precision, recall, or measured accuracy.

| Model | Vest boxes | Helmet boxes | Frames with vest |
| --- | ---: | ---: | ---: |
| SH17 | 64 | 58 | 45 |
| SafetyVision | 137 | 42 | 71 |
| Two-model ensemble | 58 | 85 | 41 |
| Additional Kaggle construction nano | 111 | 137 | 76 |
| PPE YOLO11s | 0 | 184 | 0 |

Ensemble also produced 85 uncertainty markers. At 20 seconds both existing
models still label high-visibility clothing as vests, so consensus does not
resolve the vest-only distinction. The dusty scene near 50 seconds remains
undetected by all candidates. In the clear five-worker photo, four vests are
confirmed and the fifth is uncertain. More conservative is not automatically
more accurate: labeled positives and hard negatives are needed for selection.

A separate complete-pipeline laptop benchmark (30 passes, frame at 20 seconds)
reported median 90.8 ms, mean 141.5 ms and p90 357.9 ms. This was not an isolated
hardware performance study; other laptop work and multiple runtime sessions can
cause substantial variation. Do not infer Pi FPS from it. The research grid's
larger-model comparisons also have multiple sessions resident and are not a
controlled latency ranking.

Artifacts in ppe_results/ensemble_evaluation: comparison.mp4 (five model views),
sampled JPEGs and summary.json with configuration, counts, timing and candidate
hashes. Downloaded candidate files remain local in gitignored test_models.
Neither candidate is added to robot configuration or required for the ensemble.

## Run and test

Pi, from repo root (ONNX Runtime >=1.18; same dependencies/models as parent branch):

```bash
cd Code/Server
~/ppe-venv/bin/python test_ppe_ensemble.py
~/ppe-venv/bin/python ppe_detector.py --bench --runs 50
~/ppe-venv/bin/python ppe_main.py --config ppe_config.json --no-motors --no-upload -v
```

The explicit config ensures an old local override does not disable the experiment.
Watch http://PI_IP:8080/stream.mjpg . Ctrl-C stops the process. The initial command
keeps the wheels disabled and uploads off. See ROBOT_SAFETYVISION_TEST.md for the
inherited model installation and hardware setup.

Laptop webcam (slider changes both models' display candidate thresholds):

```powershell
.\.venv-ppe\Scripts\python.exe Tools\test_ppe_webcam.py --ensemble
```

Q/Esc releases the camera. No recording is saved. The preview shows worn PPE and
vest? uncertainty markers; it does not run the compliance rule engine. Its slider
gates differ from the robot's configured per-class gates.

Reproduce the five-way research grid after downloading the two public ONNX files
to the gitignored test_models/construction_nano/best.onnx and
 test_models/ppe_yolo11s/best.onnx locations:

```powershell
.\.venv-ppe\Scripts\python.exe Tools\evaluate_ppe_ensemble.py --source test_videos\construction_laborers.mp4
```

122 detector/rule/tracker/patrol/ensemble tests pass, including one-to-one vest
matching, no score inflation, disagreement not becoming a missing-vest accusation,
canonical filtering, and preservation of original person/body-part evidence.
A saved-video simulated patrol with the ensemble entered HOLD and shut down
cleanly. None of this replaces Pi camera or labeled accuracy testing.

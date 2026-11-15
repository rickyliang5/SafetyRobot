# Alternative PPE model evaluation

Candidate: SafetyVision v1 YOLOv8-nano, author-provided ONNX.
Model card: https://huggingface.co/ayushgupta7777/safetyvision-yolov8
Download: https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/main/best.onnx
License metadata: AGPL-3.0.
The model card describes v1 as trained on Kaggle with PPE-Combined data.
The loaded ONNX metadata confirms /kaggle/working/PPE-Combined-1/data.yaml.
This is the original nano at the repository root, not the larger v2 model.

The model is about 12.2 MB, with float32 input [1,3,640,640] and raw output
[1,17,8400] (4 box channels plus 13 classes). It uses standard ONNX operators,
no custom extensions, opset 18 and IR version 10. Its initializers contain
approximately 3.03 million elements. best.onnx is self-contained; the similarly
named external data file is not referenced and is not required.

CPU runtime must support IR 10: ONNX Runtime 1.18 or newer. The existing Pi
requirement allows 1.17, which cannot load this candidate. Check the installed
runtime before testing; do not upgrade the Pi system NumPy/OpenCV camera stack.
Compatibility reference: https://onnxruntime.ai/docs/reference/compatibility.html

## Local comparison

```powershell
.\.venv-ppe\Scripts\python.exe Tools\compare_ppe_models.py --source test_videos\construction_laborers.mp4 --candidate test_models\safetyvision_nano\best.onnx
```

Both models use ONNX Runtime CPU, three intra-op threads, five warmup passes,
identical 640 input preprocessing, class-wise NMS IoU 0.45 and confidence 0.25.
Only worn helmets/vests are shown, with explicit model-specific name mapping.
This differs from the live helmet class gate of 0.40: the comparison deliberately
uses identical gates rather than comparing two differently tuned pipelines.

Reference clip: https://www.youtube.com/watch?v=s1n6nBhVzdo
420 frames sampled at stride 5 across approximately 70 seconds:

| Metric | SH17 nano | SafetyVision nano |
| --- | ---: | ---: |
| Vest boxes across sampled frames | 121 | 274 |
| Sampled frames with a vest box | 88 | 142 |
| Helmet boxes across sampled frames | 160 | 161 |
| Mean preprocess/inference/decode time, laptop CPU | 40.0 ms | 39.5 ms |
| p95 time, laptop CPU | 55.8 ms | 56.1 ms |

Timings exclude video decoding, annotation and encoding, and are not Pi timings.
They are one run, not a statistically established speed advantage. Counts are
not unique workers or measured accuracy. At 20 seconds, the candidate assigns
vest scores around 0.73-0.85 to high-visibility shirts: a visible source of
possible false positives if policy requires a distinct vest. Both models miss
PPE in the dusty scene around 50 seconds. Label representative frames before
choosing thresholds or claiming better recall/precision.

Artifacts: ppe_results/model_comparison/comparison.mp4, selected frame JPEGs,
and summary.json (class metadata, hashes, thresholds, timing and counts).

Validation: candidate ONNX graph checker passed; the comparison decoder matched
existing PpeDetector boxes/scores on frames 150, 600 and 1500. Baseline ONNX raw
outputs matched fused PyTorch raw outputs on the same preprocessed input tensor.
The existing export_onnx.py --verify image-based check reported mismatches on
workers_group.jpg. Its Ultralytics image path and the robot preprocessing path
are not identical; tensor agreement does not establish image-pipeline parity.
This remains a separate preprocessing verification concern.

## Raspberry Pi trial

Copy the candidate best.onnx, baseline exported ONNX, video, and updated Tools
scripts onto the Pi. Local test_models/test_videos are gitignored, so git pull
alone will not transfer them. Run the same comparison command with the Pi's
existing PPE Python environment, for example from the repository root:

```bash
~/ppe-venv/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"
~/ppe-venv/bin/python Tools/compare_ppe_models.py --source test_videos/construction_laborers.mp4 --candidate test_models/safetyvision_nano/best.onnx
```

This tool does not import torch or ultralytics, operate motors, or post alerts.
Its summary reports actual CPU times on the machine where it runs. It is a
lightweight Pi candidate; sustained camera capture, inference, and thermal
performance still require a hardware trial.

The original comparison above used independent models. On the
robot/safetyvision-nano branch, the robot adapter now recognizes this candidate
and retains SH17 as a separate person detector. See ROBOT_SAFETYVISION_TEST.md
for current settings and Pi commands. The dual-model robot latency is higher
than the single-model timings above. Explicit NO-* classes are kept distinct
from uncovered body parts; regression tests cover their rule handling.

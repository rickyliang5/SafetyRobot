# Visual video threshold comparison

The experimental recall-oriented defaults lower the global candidate floor
from 0.30 to 0.25, the safety-vest class threshold from 0.40 to 0.25, and the
vest presence-veto threshold from 0.35 to 0.25. Helmet, bare-body evidence,
and temporal confirmation thresholds are unchanged. On main the original required PPE was helmet only; this test branch
requires helmet and safety-vest (see ROBOT_SAFETYVISION_TEST.md). Restart the robot vision process to load changed settings;
a local configuration can override them.

Lower thresholds admit weaker candidates and can increase false positives.
They do not improve the underlying model or establish better accuracy.

Reference footage: Construction Laborers Career Video
https://www.youtube.com/watch?v=s1n6nBhVzdo

On Windows, from the repository root with the existing .venv-ppe environment:

```powershell
.\.venv-ppe\Scripts\python.exe -m pip install yt-dlp
.\.venv-ppe\Scripts\python.exe -m yt_dlp --js-runtimes node --no-playlist -f "bestvideo[ext=mp4][vcodec^=avc1][height<=720]" -o "test_videos/construction_laborers.%(ext)s" "https://www.youtube.com/watch?v=s1n6nBhVzdo"
.\.venv-ppe\Scripts\python.exe Tools\test_ppe_video.py --source test_videos\construction_laborers.mp4 --stride 5
```

Node is used by the downloader; a local video also works directly without it.
The output in ppe_results/video_thresholds contains comparison.mp4 (before
on the left, configured thresholds on the right), selected stills, and
summary.json recording thresholds and detection counts. The video has no audio.
Every fifth frame is processed by default, with the output frame rate adjusted
to preserve source timing. All sampled frames are processed unless --max-frames
is supplied. Only PPE boxes and confidence scores are displayed.

The runner uses the original PyTorch weights on CPU at input size 640 and
applies the global/class detection gates. It compares against the previous
global/vest thresholds; other class thresholds come from the selected config.
It does not execute the robot's ONNX preprocessing, compliance rules, presence
veto, tracking, or confirmation logic, and does not benchmark Raspberry Pi speed.
The veto change is covered by rule-engine checks rather than this visualization.
Use Tools/export_onnx.py --verify to separately compare ONNX and PyTorch boxes.

Counts are box detections across sampled frames, not unique workers, recall,
or precision. Review added vest boxes for correctness, inspect remaining misses,
and label representative frames before claiming an accuracy improvement. Keep a
separate unseen Pi-camera recording to evaluate settings selected using this clip.

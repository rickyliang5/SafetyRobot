# SafetyRobot

An autonomous robot that patrols a facility, spots people not wearing required PPE, and
uploads photographic evidence to an admin portal for review.

## Problem

Workplace safety remains a critical challenge across industries. Manual safety checks are sporadic and prone to human error — supervisors simply cannot be everywhere at once. Failure to wear PPE (Personal Protective Equipment) leads to severe injuries and massive regulatory fines.

---

## How It Works

1. **Autonomous Patrol** — The robot drives itself, avoiding obstacles with an ultrasonic sensor.
2. **Computer Vision** — An on-board camera feeds a YOLOv8 model trained on PPE, running locally on the Pi. No cloud, no network needed to detect.
3. **Stop and Confirm** — When it sees a person it **stops**, waits for the camera to settle, and only then starts judging. A violation must hold across multiple consecutive frames before it counts.
4. **Evidence, not alerts** — On a confirmed violation it chirps, saves a snapshot locally, and uploads it to the admin portal with the full reasoning attached. The queue survives the portal being offline and backfills later.
5. **Review** — A web portal shows every violation with its evidence, and lets a human mark each one confirmed or a false positive. Those flags produce a real precision figure.

---

## How it decides someone isn't wearing a hardhat

This is the interesting part, and it isn't what you'd first expect.

The model is a YOLOv8-nano fine-tuned on the **SH17 safety dataset** — 17 classes covering
people, body parts and PPE. Crucially, **it has no `no-helmet` class.** What it has instead is
paired classes: a bare `head` and a `helmet` are labelled *separately*.

So a violation is never a single detection. It's a geometric inference:

> attach each `head` / `helmet` box to a person by containment → a person with a bare `head`
> box and no `helmet` covering it is a violation.

That turns out to be far better than a `no-helmet` class would be. A single class only has two
states, and "didn't fire" is ambiguous between *is wearing one* and *couldn't see them*.
Separate classes give three distinct answers:

| Detected | Verdict |
|---|---|
| `helmet` covering the head | compliant |
| bare `head`, nothing covering it | **violation** — proven, the uncovered head is visible |
| neither visible (facing away, occluded) | **no call** — stays silent rather than guessing |

That third row is what keeps the portal readable. It also handles a case worth calling out: if
a helmet is detected but doesn't overlap the bare head, the person is *carrying* it, not
wearing it. Still a violation.

Compare `safety-vest`, which has no counterpart class — the only possible reasoning is "no vest
box appeared", which could equally mean someone turned around. That's weak evidence and is
held to a stricter standard (8 of 10 frames plus a clear, close, unclipped view). **Helmet is
the only item enabled by default**; vest is available once you've measured your own precision.

---

## Why it stops to look

The ultrasonic sensor and the camera are bolted to the **same pan/tilt bracket** — they
physically cannot aim independently. Rather than arbitrate between them, the two take turns:

- **Driving:** camera fixed forward and steady, inference running, only a straight-ahead
  distance reading.
- **Blocked:** stop, then sweep the bracket for bearings and turn toward open space. The camera
  swings with the sensor, so every frame captured during a sweep is **discarded**.
- **Person seen:** stop, settle for 0.3 s, *then* start accumulating evidence.

One mechanism, three payoffs. It resolves the servo conflict; it makes multi-frame agreement
meaningful (frames taken while driving show different scenes, so averaging them is nonsense);
and it removes motion blur at the exact moment a decision is made — blur being the dominant
cause of both missed detections and false alarms on a robot rolling over a hard floor.

---

## Hardware Architecture

Built on the Freenove 4WD Smart Car kit with a Raspberry Pi 5. (The original ESP32-CAM
approach is kept in `FW/` — see *Project history* below.)

| Component | Purpose |
|---|---|
| **Raspberry Pi 5 (8GB)** | On-board YOLO inference, driving, snapshot queue, live view |
| **Pi Camera** | 640×480 capture — letterboxes into the model's 640×640 with zero downscaling |
| **DC Gear Motors ×4** | Drive torque, via the kit's PCA9685 PWM driver |
| **Ultrasonic Sensor** | Obstacle detection. Shares the pan bracket with the camera |
| **Pan/Tilt Servos** | Camera aim and obstacle-bearing sweeps |
| **Buzzer + WS2812 LEDs** | Local violation signal |
| **2× 18650 cells** | Power. Expect under an hour with inference and motors both active |
| **Active Cooler** | Effectively required — sustained inference throttles a bare Pi 5 |

**No torch or ultralytics on the Pi.** The model is exported to ONNX once on a PC, and the Pi
runs `onnxruntime` alone: a 15 MB wheel instead of a 2 GB install, ~150 MB RSS instead of ~1 GB,
and it avoids ultralytics pulling `numpy>=2` into a venv where Raspberry Pi OS's `picamera2` is
compiled against numpy 1.x — a conflict that breaks camera capture with errors that look exactly
like a hardware fault.

---

## Repository layout

```
Code/Server/ppe_*.py   the patrol: detector, rule engine, tracker,
                       snapshot queue, uploader, drive state machine
Code/Server/test_*.py  96 tests, none requiring the robot
Portal/                Flask + SQLite admin portal (runs on a PC)
Tools/export_onnx.py   one-time ONNX export + parity check vs ultralytics
docs/PPE_PATROL.md     full setup, tuning and troubleshooting guide
Diagrams/              system block, detection pipeline, decision flowchart
FW/                    original ESP32-CAM firmware
Code/                  Freenove kit code (unmodified vendor files)
```

**Start here: [`docs/PPE_PATROL.md`](docs/PPE_PATROL.md)** — setup for both machines, a staged
verification plan, tuning guidance and the failure modes worth knowing about.

---

## Testing

130 tests, none of which need the robot — a simulated car and a webcam backend stand in for
hardware, so the logic that decides whether to accuse someone is fully exercised on a laptop.

```bash
cd Code/Server
python test_ppe_rules.py        # violation inference
python test_ppe_detector.py     # letterbox round-trip, class-wise NMS
python test_ppe_tracker.py      # multi-frame confirmation, de-duplication
python test_ppe_patrol.py       # drive state machine, camera handshake
cd ../../Portal && python test_portal.py   # ingest, idempotency, auth, gallery
```

---

## Project history

This started as an ESP32-CAM build; `FW/ESP32_INIT.c` and the diagrams in `Diagrams/` are from
that design. It moved to a Raspberry Pi 5 because real-time multi-class detection with
geometric reasoning over 17 classes needs considerably more compute than an ESP32 can offer —
the reasoning described above is only possible with a full detection model rather than a binary
safe/unsafe classifier.

---

## Status and limitations

**There is no map.** The chassis has no wheel encoders, no IMU and no SLAM, so coverage is
random-bounce wander rather than systematic room coverage. "Where" in a snapshot is a
configured zone label plus a timestamp, and the metadata deliberately avoids implying
localisation the robot doesn't have.

**SH17 is an industrial dataset.** A person indoors genuinely isn't wearing a hardhat, so the
model will be *correct* and the portal will fill up. Judge it on someone putting a hardhat on
and taking it off, and run the first session in `dry_run` mode.

**Privacy.** This stores identifiable photographs of people. Tell anyone in the patrol area,
set a retention policy, and keep it off the internet. The portal's auth — one shared API key
and one password over plain HTTP — is adequate for a trusted LAN and nothing more. Don't
port-forward it; use a WireGuard/Tailscale tunnel if it must be reachable remotely.

---

## Attribution and licensing

Chassis code and documentation derive from the
[Freenove 4WD Smart Car Kit](https://github.com/Freenove/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi),
released under **CC BY-NC-SA 3.0** — non-commercial use only. The vendor README is preserved at
[`docs/FREENOVE_KIT_README.md`](docs/FREENOVE_KIT_README.md).

The PPE model weights carry Ultralytics' **AGPL-3.0** metadata, which ONNX export does not
change. Fine for private and educational use; it matters if this is ever distributed or
demonstrated as a product.

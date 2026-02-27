## 📌 Problem

Workplace safety remains a critical challenge across industries. Manual safety checks are sporadic and prone to human error — supervisors simply cannot be everywhere at once. Failure to wear PPE (Personal Protective Equipment) leads to severe injuries and massive regulatory fines.


---

## 🔧 How It Works

1. **Autonomous Patrol** — The robot navigates facility floors independently using predefined paths.
2. **Computer Vision** — An on-board camera scans for human faces in the environment.
3. **Edge AI Detection** — A lightweight ML model runs locally to classify workers as **"Safe"** or **"Unsafe"**.
4. **Real-Time Alerts** — If a violation is detected (e.g., no helmet or goggles), the robot stops and issues an alert.

---

## 🏗️ Hardware Architecture

| Component | Purpose |
|---|---|
| **ESP32-CAM** | Camera module for image capture + on-device ML inference |
| **DC Gear Motors** | Drive torque for chassis movement |
| **L298N Motor Driver** | Motor bridge controller |
| **Ultrasonic Sensors** | Obstacle detection and avoidance |
| **Standard Robot Car Kit** | Chassis and structural platform |

**Total estimated cost: < $50**

---


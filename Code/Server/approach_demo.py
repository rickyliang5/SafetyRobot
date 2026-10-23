"""Drive forward slowly and stop before the object in front.

Run from Code/Server, with nothing else using the ultrasonic (stop ppe_main.py):
    python3 approach_demo.py

Reads the ultrasonic directly with lgpio and a hard per-ping timeout, so a
dead or unplugged sensor stops the robot instead of hanging the loop (gpiozero's
DistanceSensor blocks forever when no echo ever arrives).
"""
import signal
import sys
import time

import lgpio

from buzzer import Buzzer
from motor import Ordinary_Car
from servo import Servo

STOP_CM = 20.0      # brake when the object is this close
MIN_START_CM = 35.0 # refuse to start closer than this
DUTY = -1500        # negative: on this robot, positive drives AWAY from the sensor side
MAX_S = 8.0         # hard cap on drive time
TRIG, ECHO = 27, 22

car = Ordinary_Car()
car.set_motor_model(0, 0, 0, 0)


def stop_and_exit(*_):
    car.set_motor_model(0, 0, 0, 0)
    sys.exit(1)


signal.signal(signal.SIGTERM, stop_and_exit)
signal.signal(signal.SIGINT, stop_and_exit)

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, TRIG, 0)
lgpio.gpio_claim_input(h, ECHO)


def ping_cm(timeout=0.03):
    """One reading in cm, or None if the sensor didn't answer in time."""
    lgpio.gpio_write(h, TRIG, 1); time.sleep(0.00001); lgpio.gpio_write(h, TRIG, 0)
    t0 = time.monotonic()
    while not lgpio.gpio_read(h, ECHO):
        if time.monotonic() - t0 > timeout:
            return None
    start = time.monotonic()
    while lgpio.gpio_read(h, ECHO):
        if time.monotonic() - start > timeout:
            return None
    return (time.monotonic() - start) * 17150


def distance_cm():
    """Median of the pings that answered out of 3; None only if all 3 failed.

    A single dropped echo is normal while driving (vibration, glancing angle),
    so one miss must not count as a dead sensor.
    """
    vals = []
    for _ in range(3):
        v = ping_cm()
        if v is not None:
            vals.append(v)
        time.sleep(0.01)
    if not vals:
        return None
    return sorted(vals)[len(vals) // 2]


bz = Buzzer()
servo = Servo()
servo.set_servo_pwm('0', 90)   # pan straight ahead
servo.set_servo_pwm('1', 90)   # tilt level


def chirp(t=0.08):
    bz.set_state(True); time.sleep(t); bz.set_state(False)


try:
    time.sleep(0.5)
    d = distance_cm()
    if d is None:
        sys.exit("ultrasonic not answering -- not driving")
    print(f"object ahead at {d:.1f} cm", flush=True)
    if d < MIN_START_CM:
        sys.exit(f"object is closer than {MIN_START_CM:.0f} cm -- move it back and retry")

    for _ in range(3):                      # 3-second countdown
        chirp(); time.sleep(0.92)

    t0 = time.monotonic()
    car.set_motor_model(DUTY, DUTY, DUTY, DUTY)
    misses = 0
    while True:
        d = distance_cm()
        if d is None:
            misses += 1
            if misses >= 3:                  # worst case ~0.3 s (~15 cm) with no echo at all
                print("sensor stopped answering -- stopping", flush=True)
                break
            continue
        misses = 0
        if d <= STOP_CM:
            print(f"object at {d:.1f} cm -- stopping", flush=True)
            break
        if time.monotonic() - t0 > MAX_S:
            print(f"{MAX_S:.0f}s cap reached at {d:.1f} cm -- stopping", flush=True)
            break
    car.set_motor_model(0, 0, 0, 0)
    drove = time.monotonic() - t0

    time.sleep(0.5)                          # let it settle, then measure
    final = distance_cm()
    print(f"drove {drove:.1f}s; stopped "
          f"{'?' if final is None else f'{final:.1f}'} cm from the object", flush=True)
    chirp(0.3)
finally:
    car.set_motor_model(0, 0, 0, 0)
    lgpio.gpiochip_close(h)

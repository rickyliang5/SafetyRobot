"""One stationary PPE inspection after a short obstacle course."""

from __future__ import annotations

import math
import time

from ppe_patrol import BLOCKED_SCAN, HOLD, STOPPED, WANDER, PatrolDrive

DETOUR = "DETOUR"
RETURN_HEADING = "RETURN_HEADING"


class DemoDrive(PatrolDrive):
    """Use sonar to select a side, then make a bounded, timed detour."""

    pivot_s = 0.6
    detour_s = 1.8
    timeout_s = 35.0

    def __init__(self, car, shared, cfg) -> None:
        super().__init__(car, shared, cfg)
        self.error: str | None = None
        self.detoured = False
        self._direction = 0
        self._started_at: float | None = None
        self._last_frame_count = 0
        self._last_frame_at = 0.0

    def _tick(self) -> None:
        now = time.monotonic()
        try:
            if self._started_at is None:
                self._started_at = now
                self._last_battery_check = float("-inf")
                self._brake()
                self.car.set_servo_pwm("0", self.p.pan_center)
                self._phase_since = now
                print("[demo] Driving straight toward the obstacle")
            self._check_battery(now)
            state = self.shared.snapshot()
            if state["inspection_id"]:
                self._enter(STOPPED)
                self._release_hold()
                self._brake()
                return
            if state["frames"] != self._last_frame_count:
                self._last_frame_count = state["frames"]
                self._last_frame_at = now
            if now - self._last_frame_at > 3:
                raise RuntimeError("camera has stopped delivering detections")
            if now - self._started_at >= self.timeout_s:
                raise RuntimeError("course timed out before a complete PPE inspection")
            if self.state == WANDER:
                if now - self._phase_since >= self._sonar_settle_s:
                    self._tick_wander(now)
            elif self.state == BLOCKED_SCAN:
                self._tick_blocked_scan(now)
            elif self.state == DETOUR:
                self._tick_detour(now)
            elif self.state == RETURN_HEADING:
                self._tick_return(now)
            elif self.state == HOLD:
                self._tick_hold(now)
        except Exception as exc:
            self.error = str(exc)
            self._stop_event.set()
            self._release_hold()
            self._brake()
            print(f"[demo] Stopped: {exc}")

    def _tick_wander(self, now: float) -> None:
        distance = self._read_distance()
        state = self.shared.snapshot()
        if self.detoured and state["request_hold"]:
            self._enter(HOLD)
            self._set_steady(False)
            print("[demo] Person found. Stopped to inspect PPE.")
        elif distance < self.p.stop_distance_cm:
            if self.detoured:
                raise RuntimeError("another obstacle blocked the route after the detour")
            self._enter(BLOCKED_SCAN)
            self._set_steady(False)
            self._set_pan(self.p.pan_left)
            print("[demo] Obstacle detected. Checking left and right clearance.")
        else:
            self.car.set_motor_model(*([self.p.cruise_duty] * 4))

    def _tick_blocked_scan(self, now: float) -> None:
        elapsed = now - self._phase_since
        if self._phase == 0 and elapsed >= self._sonar_settle_s:
            self._scan_readings["left"] = self._read_distance(update_shared=False)
            self._set_pan(self.p.pan_right)
            self._next_phase()
        elif self._phase == 1 and elapsed >= self._sonar_settle_s:
            right = self._read_distance(update_shared=False)
            left = self._scan_readings["left"]
            if max(left, right) < self.p.stop_distance_cm:
                raise RuntimeError("neither side of the obstacle is clear")
            self._direction = -1 if left >= right else 1
            self._set_pan(self.p.pan_center)
            self._next_phase()
        elif self._phase == 2 and elapsed >= self._sonar_settle_s:
            self._pivot(self._direction)
            self._next_phase()
        elif self._phase == 3 and elapsed >= self.pivot_s:
            self._brake()
            self._enter(DETOUR)

    def _tick_detour(self, now: float) -> None:
        if self._read_distance() < self.p.stop_distance_cm:
            raise RuntimeError("the path around the obstacle is blocked")
        if self._phase == 0:
            self.car.set_motor_model(*([self.p.cruise_duty] * 4))
            self._next_phase()
        elif now - self._phase_since >= self.detour_s:
            self._brake()
            self._enter(RETURN_HEADING)
            self._pivot(-self._direction)

    def _tick_return(self, now: float) -> None:
        if now - self._phase_since >= self.pivot_s:
            self._brake()
            self.detoured = True
            self._set_steady(True)
            self._enter(WANDER)
            print("[demo] Detour complete. Continuing forward to find the person.")

    def _tick_hold(self, now: float) -> None:
        self._brake()
        if self._phase == 0 and now - self._state_since >= 0.3:
            with self.shared.lock:
                self.shared.camera_steady = True
                self.shared.hold_granted = True
            self._next_phase()

    def _pivot(self, direction: int) -> None:
        duty = direction * self.p.pivot_duty
        self.car.set_motor_model(duty, duty, -duty, -duty)

    @property
    def _sonar_settle_s(self) -> float:
        # The sensor reports a median of nine samples at 60 ms per sample.
        return max(self.p.pan_settle_s, 0.8)

    def _read_distance(self, update_shared: bool = True) -> float:
        distance = self.car.get_distance()
        if distance is None or not math.isfinite(distance) or distance <= 0:
            raise RuntimeError("ultrasonic sensor did not return a usable distance")
        if update_shared:
            with self.shared.lock:
                self.shared.sonic_center_cm = distance
        return distance

    def _check_battery(self, now: float) -> None:
        if now - self._last_battery_check < 2:
            return
        volts = self.car.battery_voltage()
        if not math.isfinite(volts) or not self.p.min_battery_v <= volts <= 9:
            raise RuntimeError(f"battery reading {volts:.2f} V is outside the demo range")
        self._last_battery_check = now
        with self.shared.lock:
            self.shared.battery_v = volts

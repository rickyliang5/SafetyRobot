"""Run both demo cases with simulated wheels, sonar and PPE observations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ppe_config import load_config
from ppe_demo import DemoDrive
from ppe_patrol import BLOCKED_SCAN, HOLD, STOPPED, PatrolVision, SharedState, SimulatedCar
from ppe_rules import RuleEngine
from ppe_snapshot import SnapshotStore
from ppe_tracker import PatrolVisionState
from ppe_uploader import from_config
from test_ppe_tracker import FRAME, bare_head_frame, det, helmeted_frame


class DemoTests(unittest.TestCase):
    def run_case(self, wearing_ppe: bool) -> None:
        cfg = load_config()
        car = SimulatedCar()
        shared = SharedState()
        drive = DemoDrive(car, shared, cfg)
        now = 100.0

        def distance():
            if drive.state == BLOCKED_SCAN:
                return 140.0 if shared.pan_angle == cfg.patrol.pan_left else 65.0
            if not drive.detoured and now >= 101.8 and drive.state == "WANDER":
                return 25.0
            return 150.0

        car.get_distance = distance
        face = det("face", (301., 120., 340., 166.))
        detections = bare_head_frame() + [face]
        if wearing_ppe:
            detections = helmeted_frame() + [
                face, det("glasses", (299., 128., 342., 140.)),
                det("safety-vest", (275., 175., 360., 310.)),
            ]
        observations = RuleEngine(cfg.rules, cfg.model).evaluate_frame(detections, FRAME)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(Path(directory))
            vision = PatrolVision(None, shared, cfg, store=store, demo=True)
            vision.vision_state = PatrolVisionState(cfg.temporal, cfg.cooldown)
            with patch("ppe_demo.time.monotonic", side_effect=lambda: now):
                for tick in range(120):
                    now = 100 + tick * .2
                    shared.frames_total += 1
                    shared.request_hold = drive.detoured
                    drive._tick()
                    if shared.may_record():
                        for event in vision.vision_state.process(
                            observations, now, frame, report_cases=True, require_complete=True,
                        ):
                            self.assertEqual(drive.state, HOLD)
                            self.assertEqual(car.motor, (0, 0, 0, 0))
                            vision._handle_violation(event, frame, FRAME)
                    if shared.inspection_id:
                        drive._tick()
                        break
            self.assertIsNone(drive.error)
            self.assertTrue(drive.detoured)
            self.assertEqual(drive.state, STOPPED)
            self.assertEqual(car.motor, (0, 0, 0, 0))
            motors = [args for kind, args in car.commands if kind == "motor"]
            self.assertIn((cfg.patrol.cruise_duty,) * 4, motors)
            turn = cfg.patrol.pivot_duty
            self.assertLess(motors.index((-turn, -turn, turn, turn)),
                            motors.index((turn, turn, -turn, -turn)))
            event, = store.iter_pending()
            payload = event.load_meta()
            self.assertEqual(payload["event_id"], shared.inspection_id)
            self.assertEqual(payload["status"], "compliant" if wearing_ppe else "non_compliant")
            self.assertEqual(set(payload["missing_ppe"]),
                             set() if wearing_ppe else set(cfg.rules.required_ppe))
            with patch("requests.post") as post:
                post.return_value.status_code = 201
                self.assertEqual(from_config(cfg, store).post_event(event), (True, False))
                self.assertIn("/api/v1/violations", post.call_args.args[0])

    def test_wearing_required_ppe(self):
        self.run_case(wearing_ppe=True)

    def test_missing_all_required_ppe(self):
        self.run_case(wearing_ppe=False)

    def test_unpowered_robot_does_not_move(self):
        car = SimulatedCar()
        car.battery_voltage = lambda: 0.0
        drive = DemoDrive(car, SharedState(frames_total=1), load_config())
        drive._tick()
        self.assertIn("battery", drive.error)
        self.assertFalse(any(any(args) for kind, args in car.commands if kind == "motor"))

    def test_sonar_failure_stops_the_wheels(self):
        car = SimulatedCar()
        car.get_distance = lambda: None
        drive = DemoDrive(car, SharedState(frames_total=1), load_config())
        with patch("ppe_demo.time.monotonic", return_value=100.0) as clock:
            drive._tick()
            clock.return_value = 101.0
            drive._tick()
        self.assertIn("ultrasonic", drive.error)
        self.assertEqual(car.motor, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

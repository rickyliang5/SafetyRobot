"""Tests for the face crop and its handling in the snapshot queue.

Run: python test_ppe_snapshot.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ppe_classes import CLASS_IDS
from ppe_rules import PersonObservation
from ppe_snapshot import SnapshotStore, crop_face
from ppe_types import Detection

PERSON = (250.0, 100.0, 390.0, 400.0)
FACE = (300.0, 120.0, 340.0, 170.0)


def det(name, box, conf=0.9):
    return Detection(cls_id=CLASS_IDS[name], name=name, conf=conf, xyxy=box)


def frame():
    return np.full((480, 640, 3), 80, dtype=np.uint8)


def test_crop_prefers_the_face_and_pads_it():
    obs = PersonObservation(det("person", PERSON), attached={"face": [det("face", FACE)]})
    crop = crop_face(frame(), obs, min_height=1)
    # 40x50 face + 35% padding each side -> about 68x85 (pixel rounding)
    h, w = crop.shape[:2]
    assert abs(h - 85) <= 1 and abs(w - 68) <= 1, crop.shape


def test_crop_falls_back_to_top_of_person():
    obs = PersonObservation(det("person", PERSON))
    crop = crop_face(frame(), obs, min_height=1)
    assert crop is not None and crop.shape[0] < 120, "should be the top fifth, padded"


def test_small_crop_is_upscaled():
    obs = PersonObservation(det("person", PERSON), attached={"face": [det("face", FACE)]})
    assert crop_face(frame(), obs, min_height=160).shape[0] == 160


def test_crop_clamped_at_frame_edge():
    edge = (0.0, 0.0, 30.0, 30.0)
    obs = PersonObservation(det("person", (0.0, 0.0, 100.0, 300.0)),
                            attached={"face": [det("face", edge)]})
    crop = crop_face(frame(), obs, min_height=1)
    assert crop is not None and crop.shape[0] > 0


def test_face_written_moved_and_survives_prune():
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(tmp)
        payload = {"event_id": "abc123"}
        store.write_event(frame(), payload, face_bgr=frame()[:100, :80])
        assert payload["face_image"] is True
        [event] = list(store.iter_pending())
        assert event.face.is_file()

        # prune() reaps .jpg files with no matching .json; the face crop's
        # .json is <base>.json, not <base>_face.json, so it must be kept.
        old = time.time() - 600
        os.utime(event.face, (old, old))
        store.prune()
        assert event.face.is_file(), "prune deleted a live face crop"

        store.mark_sent(event)
        assert (Path(tmp) / "sent" / event.face.name).is_file()
        assert not event.face.exists()


def test_no_face_flag_when_none_given():
    with tempfile.TemporaryDirectory() as tmp:
        payload = {"event_id": "noface"}
        SnapshotStore(tmp).write_event(frame(), payload)
        assert payload["face_image"] is False


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("PPE snapshot / face crop tests\n")
    sys.exit(_run())

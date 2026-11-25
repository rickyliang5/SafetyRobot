"""Tests for tracking, temporal confirmation and de-duplication.

Deterministic: time is injected, never read from the clock, so these cover the
cooldown and rate-limit logic without a single sleep.

Run: python test_ppe_tracker.py
"""

from __future__ import annotations

import sys

from ppe_classes import CLASS_IDS
from ppe_config import CooldownConfig, ModelConfig, RulesConfig, TemporalConfig
from ppe_rules import RuleEngine
from ppe_tracker import PatrolVisionState, PersonTracker, ViolationConfirmer
from ppe_types import Detection

FRAME = (640, 480)

NEAR_PERSON = (250.0, 100.0, 390.0, 400.0)
NEAR_HEAD = (295.0, 110.0, 345.0, 170.0)
NEAR_HELMET = (292.0, 100.0, 348.0, 150.0)


def det(name, box, conf=0.9):
    return Detection(cls_id=CLASS_IDS[name], name=name, conf=conf, xyxy=box)


def rules_engine(required=None):
    return RuleEngine(RulesConfig(required_ppe=required or ["helmet"]), ModelConfig())


def bare_head_frame(person=NEAR_PERSON, head=NEAR_HEAD):
    return [det("person", person, 0.91), det("head", head, 0.88)]


def helmeted_frame(person=NEAR_PERSON):
    return [det("person", person, 0.91), det("helmet", NEAR_HELMET, 0.80)]


def observe(dets, required=None):
    return rules_engine(required).evaluate_frame(dets, FRAME)


def temporal(**kw):
    base = dict(window_m=10, positive_n=5, absence_n=8, min_track_hits=4)
    base.update(kw)
    return TemporalConfig(**base)


def cooldown(**kw):
    base = dict(
        per_person_s=90.0,
        global_min_interval_s=8.0,
        max_per_hour=60,
        reid_similarity=0.85,
        reid_window_s=120.0,
    )
    base.update(kw)
    return CooldownConfig(**base)


def state(temporal_kw=None, cooldown_kw=None):
    return PatrolVisionState(
        temporal(**(temporal_kw or {})), cooldown(**(cooldown_kw or {}))
    )


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_same_person_keeps_one_track_id():
    tracker = PersonTracker()
    ids = set()
    for i in range(6):
        drift = (250.0 + i * 3, 100.0, 390.0 + i * 3, 400.0)
        obs = observe(bare_head_frame(person=drift))
        pairs = tracker.update(obs, now=float(i))
        ids.add(pairs[0][0].track_id)
    assert ids == {1}, f"a slowly-walking person must stay one track, got {ids}"


def test_two_people_get_two_ids():
    tracker = PersonTracker()
    far = (40.0, 100.0, 180.0, 400.0)
    obs = observe(bare_head_frame() + [det("person", far, 0.90)])
    pairs = tracker.update(obs, now=0.0)
    assert len({t.track_id for t, _ in pairs}) == 2


def test_track_ages_out_after_max_misses():
    tracker = PersonTracker(max_misses=3)
    tracker.update(observe(bare_head_frame()), now=0.0)
    assert len(tracker.tracks) == 1
    for i in range(4):
        tracker.update([], now=float(i + 1))
    assert tracker.tracks == {}


def test_track_survives_a_brief_miss():
    tracker = PersonTracker(max_misses=3)
    tracker.update(observe(bare_head_frame()), now=0.0)
    tracker.update([], now=1.0)
    pairs = tracker.update(observe(bare_head_frame()), now=2.0)
    assert pairs[0][0].track_id == 1
    assert pairs[0][0].misses == 0


def test_fast_motion_matches_by_centroid():
    """A jump too big for IoU should still match, not spawn a new id."""
    tracker = PersonTracker()
    tracker.update(observe(bare_head_frame()), now=0.0)
    moved = (310.0, 100.0, 450.0, 400.0)  # shifted 60px, IoU ~0.44 -> still ok
    far_jump = (355.0, 100.0, 495.0, 400.0)  # shifted 105px, IoU below thresh
    tracker.update(observe(bare_head_frame(person=moved)), now=1.0)
    pairs = tracker.update(observe(bare_head_frame(person=far_jump)), now=2.0)
    assert pairs[0][0].track_id == 1, "centroid fallback should hold identity"


# --------------------------------------------------------------------------
# N-of-M confirmation
# --------------------------------------------------------------------------


def test_single_bad_frame_does_not_fire():
    """The core anti-false-positive guarantee."""
    st = state()
    events = st.process(observe(bare_head_frame()), now=0.0)
    assert events == []


def test_fires_after_enough_agreeing_frames():
    st = state()
    fired = None
    for i in range(10):
        events = st.process(observe(bare_head_frame()), now=float(i))
        if events:
            fired = (i, events[0])
            break
    assert fired is not None, "5-of-10 agreement should eventually confirm"
    i, event = fired
    assert i == 4, f"expected confirmation on the 5th frame, got the {i + 1}th"
    assert event.items == ["helmet"]
    assert event.frames_agree["helmet"] == "5/5"
    assert event.track_id == 1


def test_min_track_hits_gates_confirmation():
    """Even unanimous frames wait for the track to establish itself."""
    st = state(temporal_kw={"positive_n": 2, "min_track_hits": 6})
    for i in range(5):
        assert st.process(observe(bare_head_frame()), now=float(i)) == []
    events = st.process(observe(bare_head_frame()), now=5.0)
    assert events, "should fire once min_track_hits is reached"


def test_compliant_person_never_fires():
    st = state()
    for i in range(20):
        assert st.process(observe(helmeted_frame()), now=float(i)) == []


def test_presence_in_the_window_vetoes():
    """One confident helmet sighting clears the whole window.

    Someone who turns their head and briefly shows a helmet should not be
    reported, even if most frames looked like a bare head.
    """
    st = state()
    for i in range(4):
        st.process(observe(bare_head_frame()), now=float(i))
    st.process(observe(helmeted_frame()), now=4.0)
    events = st.process(observe(bare_head_frame()), now=5.0)
    assert events == [], "a PRESENT frame must veto accumulated evidence"


def test_unknown_frames_do_not_accumulate():
    """A person we can't assess is not evidence of anything."""
    st = state()
    for i in range(12):
        events = st.process(observe([det("person", NEAR_PERSON, 0.91)]), now=float(i))
        assert events == []


def test_absence_only_item_needs_a_stricter_majority():
    """safety-vest at 8-of-10 vs helmet at 5-of-10."""
    st = state()
    first_fire = None
    for i in range(12):
        events = st.process(
            observe([det("person", NEAR_PERSON, 0.91)], required=["safety-vest"]),
            now=float(i),
        )
        if events:
            first_fire = i
            break
    assert first_fire == 7, f"expected the 8th frame, got the {(first_fire or 0) + 1}th"


def test_moving_frames_are_not_recorded():
    """record=False keeps identity but refuses to accumulate evidence.

    This is what makes N-of-M honest: frames taken while the robot drives or
    pans show different scenes and must not count toward agreement.
    """
    st = state()
    for i in range(20):
        events = st.process(observe(bare_head_frame()), now=float(i), record=False)
        assert events == []
    assert len(st.tracker.tracks) == 1, "identity should still be maintained"


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------


def test_one_person_reports_once_then_cools_down():
    """The anti-spam guarantee: standing in view must not yield 100 uploads."""
    st = state()
    events = []
    for i in range(60):
        events.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(events) == 1, f"expected a single report, got {len(events)}"


def test_reports_again_after_the_cooldown_expires():
    """A still-uncompliant person is re-reported once the hold lapses."""
    st = state(cooldown_kw={"per_person_s": 20.0})
    first = []
    for i in range(10):
        first.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(first) == 1
    assert st.tracker.tracks[1].item_cooldown["helmet"] == 24.0

    # Fires once more as soon as the cooldown lapses, then holds again. The
    # window stops just short of the NEXT cooldown expiry (24 + 20 = 44 at the
    # earliest), so exactly one report is correct here.
    second = []
    for i in range(30, 44):
        second.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(second) == 1, f"expected one repeat report, got {len(second)}"


def test_persisting_violation_reports_once_per_cooldown():
    """Rate of repeat reporting is governed by per_person_s, not frame rate.

    Someone standing in view without a helmet for two minutes should produce a
    handful of periodic reports, not hundreds -- and not exactly one either,
    since an ongoing violation is worth re-surfacing.
    """
    st = state(cooldown_kw={"per_person_s": 30.0})
    events = []
    for i in range(121):  # ~2 minutes at 1 fps
        events.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert 3 <= len(events) <= 5, (
        f"expected roughly one report per 30s cooldown, got {len(events)}"
    )


def test_global_min_interval_throttles_separate_people():
    """Two different people violating at once must not both fire instantly."""
    st = state(cooldown_kw={"global_min_interval_s": 30.0})
    a = (40.0, 100.0, 180.0, 400.0)
    a_head = (85.0, 110.0, 135.0, 170.0)
    emitted = []
    for i in range(10):
        dets = bare_head_frame() + [
            det("person", a, 0.90),
            det("head", a_head, 0.88),
        ]
        emitted.extend(st.process(observe(dets), now=float(i)))
    assert len(emitted) == 1, "global interval should hold the second report back"


def test_max_per_hour_is_a_hard_cap():
    st = state(
        temporal_kw={"positive_n": 1, "min_track_hits": 1},
        cooldown_kw={"per_person_s": 0.1, "global_min_interval_s": 0.0, "max_per_hour": 3},
    )
    emitted = []
    for i in range(40):
        # Reset only the TRACKER, so every frame looks like a brand-new person
        # (worst-case id churn). Resetting the whole state would also clear the
        # rate limiter, which is the thing under test.
        st.tracker.reset()
        emitted.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(emitted) == 3, f"cap is 3/hour, emitted {len(emitted)}"


def test_second_item_goes_on_the_persons_next_report():
    """One card per person: a newly-confirmed item waits for the cooldown."""
    st = state(
        temporal_kw={"positive_n": 2, "min_track_hits": 2},
        cooldown_kw={"per_person_s": 20.0, "global_min_interval_s": 0.0},
    )
    req = ["helmet", "gloves"]
    gloved = bare_head_frame() + [det("gloves", (255.0, 250.0, 300.0, 300.0), 0.80)]
    bare = bare_head_frame() + [det("hands", (255.0, 250.0, 300.0, 300.0), 0.80)]
    events = []
    for i in range(6):
        events += st.process(observe(gloved, required=req), now=float(i))
    assert [e.items for e in events] == [["helmet"]]
    # Gloves come off a few seconds later: no second card yet...
    for i in range(6, 20):
        events += st.process(observe(bare, required=req), now=float(i))
    assert len(events) == 1, "a second card for the same person within the cooldown"
    # ...it's on their next report, once the cooldown lapses.
    for i in range(20, 30):
        events += st.process(observe(bare, required=req), now=float(i))
    assert len(events) == 2 and "gloves" in events[1].items


# --------------------------------------------------------------------------
# evidence selection
# --------------------------------------------------------------------------


def test_best_frame_is_the_most_confident_one():
    """The snapshot should be the clearest frame, not the one that tipped N."""
    st = state()
    confs = [0.60, 0.95, 0.62, 0.64, 0.66]
    for i, c in enumerate(confs):
        dets = [det("person", NEAR_PERSON, c), det("head", NEAR_HEAD, 0.88)]
        events = st.process(observe(dets), now=float(i))
        if events:
            assert abs(events[0].observation.person.conf - 0.95) < 1e-6, (
                "evidence should come from the highest-confidence frame"
            )
            return
    raise AssertionError("never confirmed")


def test_event_json_round_trips():
    st = state()
    for i in range(10):
        events = st.process(observe(bare_head_frame()), now=float(i))
        if events:
            payload = events[0].to_json()
            assert payload["missing_ppe"] == ["helmet"]
            assert payload["track_id"] == 1
            assert "person_bbox" in payload and len(payload["person_bbox"]) == 4
            assert payload["items"]["helmet"]["status"] == "MISSING_POSITIVE"
            return
    raise AssertionError("never confirmed")


# --------------------------------------------------------------------------
# confirmer unit-level
# --------------------------------------------------------------------------


def test_confirmer_counts_are_reported_accurately():
    tracker = PersonTracker(window=10)
    confirmer = ViolationConfirmer(temporal(), cooldown())
    for i in range(5):
        pairs = tracker.update(observe(bare_head_frame()), now=float(i))
        track = pairs[0][0]
    positive, absent, present, total = track.counts("helmet")
    assert (positive, absent, present, total) == (5, 0, 0, 5)
    assert confirmer.confirmed_items(track) == {"helmet": "5/5"}


def test_window_slides_and_drops_old_verdicts():
    tracker = PersonTracker(window=4)
    for i in range(10):
        pairs = tracker.update(observe(bare_head_frame()), now=float(i))
    track = pairs[0][0]
    assert track.counts("helmet")[3] == 4, "history must be capped at the window"


# --------------------------------------------------------------------------
# time-based confirmation, compliant reports, steady display
# --------------------------------------------------------------------------


def test_violation_waits_for_min_duration():
    # 10 agreeing frames 0.1 s apart is a burst, not a sustained condition.
    vs = state({"min_duration_s": 2.0})
    events = []
    for i in range(10):
        events += vs.process(observe(bare_head_frame()), now=i * 0.1)
    assert events == [], "a 1 s burst must not convict"
    # At the robot's real ~4 fps the same 10-frame window covers 2+ seconds.
    t = 1.0
    for _ in range(10):
        t += 0.3
        events += vs.process(observe(bare_head_frame()), now=t)
    assert len(events) == 1 and events[0].items == ["helmet"]
    assert events[0].status == "non_compliant"


def test_compliant_reported_once_per_person():
    vs = state({"compliant_n": 5})
    events = []
    for i in range(20):
        events += vs.process(observe(helmeted_frame()), now=float(i))
    assert len(events) == 1, f"one compliant report expected, got {len(events)}"
    assert events[0].status == "compliant"
    assert events[0].items == [] and events[0].present == ["helmet"]


def test_compliant_off_by_default():
    vs = state()
    events = []
    for i in range(20):
        events += vs.process(observe(helmeted_frame()), now=float(i))
    assert events == []


def test_one_missing_frame_blocks_compliant():
    vs = state({"compliant_n": 5})
    frames = [helmeted_frame()] * 4 + [bare_head_frame()] + [helmeted_frame()] * 4
    events = []
    for i, f in enumerate(frames):
        events += vs.process(observe(f), now=float(i))
    assert events == [], "a missing frame inside the window must veto compliant"


def test_display_holds_through_flicker():
    vs = state({"compliant_n": 5})
    for i in range(8):
        obs = observe(helmeted_frame())
        vs.process(obs, now=float(i))
    assert vs.display_for(obs)[0] == {"helmet": "OK"}
    # A single bad frame must not flip the label.
    obs = observe(bare_head_frame())
    vs.process(obs, now=8.0)
    assert vs.display_for(obs)[0] == {"helmet": "OK"}


def test_display_starts_checking_then_missing():
    vs = state()
    obs = observe(bare_head_frame())
    vs.process(obs, now=0.0)
    assert vs.display_for(obs)[0] == {"helmet": "CHECKING"}
    for i in range(1, 8):
        obs = observe(bare_head_frame())
        vs.process(obs, now=float(i))
    assert vs.display_for(obs)[0] == {"helmet": "MISSING"}


# --------------------------------------------------------------------------
# one report per person, with a photo that matches it
# --------------------------------------------------------------------------

NEAR_FACE = (300.0, 125.0, 340.0, 165.0)
NEAR_GLASSES = (300.0, 135.0, 340.0, 150.0)


def bare_head_and_face():
    return bare_head_frame() + [det("face", NEAR_FACE, 0.85)]


def test_waits_and_reports_all_missing_items_together():
    """The split-card bug: helmet confirmed first, glasses a little later."""
    st = state({"decide_timeout_s": 30.0, "compliant_n": 5})
    req = ["helmet", "eye-protection"]
    events = []
    # Glasses look fine at first, so only the helmet is decided...
    frames = [bare_head_frame() + [det("face", NEAR_FACE, 0.85), det("glasses", NEAR_GLASSES, 0.8)]] * 3
    # ...then they come off and the face is bare.
    frames += [bare_head_and_face()] * 12
    for i, f in enumerate(frames):
        events += st.process(observe(f, required=req), now=float(i))
    assert len(events) == 1, f"one report per person expected, got {len(events)}"
    assert events[0].items == ["eye-protection", "helmet"]


def test_long_watched_person_still_gets_one_card():
    """Track 23's bug: a person watched for a while, then hat off, then
    glasses off a few seconds later, got a separate card per item."""
    st = state({"decide_timeout_s": 5.0}, {"per_person_s": 1000.0})
    req = ["helmet", "eye-protection"]
    face, glasses = det("face", NEAR_FACE, 0.85), det("glasses", NEAR_GLASSES, 0.8)
    all_on = helmeted_frame() + [face, glasses]
    hat_off = bare_head_frame() + [face, glasses]
    both_off = bare_head_frame() + [face]
    timeline = [all_on] * 40 + [hat_off] * 3 + [both_off] * 20
    events = []
    for i, frame in enumerate(timeline):
        events += st.process(observe(frame, required=req), now=float(i))
    assert len(events) == 1, f"expected one card, got {[e.items for e in events]}"
    assert "helmet" in events[0].items


def test_reports_after_timeout_with_unsure_items():
    """An item that never resolves must not hold a violation back forever."""
    st = state({"decide_timeout_s": 4.0})
    req = ["helmet", "eye-protection"]   # no face box -> glasses stay unsure
    events = []
    for i in range(12):
        events += st.process(observe(bare_head_frame(), required=req), now=float(i))
    assert len(events) == 1
    assert events[0].items == ["helmet"] and events[0].unsure == ["eye-protection"]


def test_no_compliant_card_straight_after_a_violation():
    st = state({"compliant_n": 5})
    events = []
    for i in range(10):
        events += st.process(observe(bare_head_frame()), now=float(i))
    for i in range(10, 25):   # puts the helmet on within the cooldown
        events += st.process(observe(helmeted_frame()), now=float(i))
    assert [e.status for e in events] == ["non_compliant"]


def test_photo_agrees_with_the_verdict():
    """The stale-photo bug: a clear early frame must not illustrate a later verdict."""
    st = state()
    events = []
    # A very confident frame WITH a helmet, then the helmet comes off.
    events += st.process(observe(
        [det("person", NEAR_PERSON, 0.99), det("helmet", NEAR_HELMET, 0.80)]
    ), now=0.0)
    for i in range(1, 15):
        events += st.process(observe(bare_head_frame()), now=float(i))
    assert events, "should have confirmed"
    assert events[0].observation.items["helmet"].is_missing, (
        "the report photo shows the helmet on"
    )
    assert events[0].frame is None  # no camera frames in these tests


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
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
    print("PPE tracker / confirmation tests\n")
    sys.exit(_run())

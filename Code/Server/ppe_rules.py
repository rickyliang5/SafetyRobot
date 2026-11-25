"""Per-frame PPE violation inference.

This is where "a list of boxes" becomes "that person isn't wearing a helmet",
and it is the part of the system most likely to be wrong in a way nobody
notices, so it is pure logic with no camera, model or hardware imports and it
has a real test suite.

The model (SH17) has no "no-helmet" class. What it has is paired classes: it
labels a bare `head` and a `helmet` separately, bare `hands` and `gloves`
separately, and so on. That gives two very different grades of evidence:

  POSITIVE  we can see an uncovered body part -> the PPE is genuinely absent.
            Trustworthy. This is what makes helmet enforcement work.

  ABSENCE   no PPE box appeared, and there is no body-part class that would
            prove it either way (`safety-vest` is the only such item). Weak:
            the person may have their back turned, be half out of frame, or be
            occluded. Held to a much stricter standard.

Three mechanisms keep the asymmetry honest:

  1. Presence veto -- one confident PPE detection clears the item outright.
     Positive evidence always beats inference.
  2. Separate thresholds -- `class_conf` to be a detection, the higher
     `evidence_conf` to be used as proof of a violation.
  3. View gate -- absence evidence only counts when we can demonstrably see
     enough of the person to judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ppe_classes import PPE_ITEMS, PERSON
from ppe_config import ModelConfig, RulesConfig
from ppe_types import (
    Detection,
    overlap_frac,
    pad_box,
    touches_edge,
    vertical_band,
)

# PRESENT            PPE seen on this person.
# MISSING_POSITIVE   uncovered body part seen -> PPE genuinely absent.
# MISSING_ABSENT     no PPE box, no proof either way -> weak signal.
# UNKNOWN            cannot judge: ambiguous ownership, or view too poor.
ItemStatus = Literal["PRESENT", "MISSING_POSITIVE", "MISSING_ABSENT", "UNKNOWN"]


@dataclass(frozen=True)
class ItemEvidence:
    """Why we reached a verdict about one PPE item on one person."""

    status: ItemStatus
    support: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def is_missing(self) -> bool:
        return self.status in ("MISSING_POSITIVE", "MISSING_ABSENT")

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "support": {k: round(float(v), 4) for k, v in self.support.items()},
            "reason": self.reason,
        }


@dataclass
class PersonObservation:
    """One person in one frame, with every PPE item assessed."""

    person: Detection
    items: dict[str, ItemEvidence] = field(default_factory=dict)
    attached: dict[str, list[Detection]] = field(default_factory=dict)
    view_ok: bool = False
    truncated: bool = False
    height_frac: float = 0.0

    def missing_items(self) -> list[str]:
        return [name for name, ev in self.items.items() if ev.is_missing]

    def to_json(self) -> dict:
        return {
            "person_bbox": [round(float(v), 1) for v in self.person.xyxy],
            "person_conf": round(float(self.person.conf), 4),
            "view_ok": self.view_ok,
            "truncated": self.truncated,
            "height_frac": round(self.height_frac, 3),
            "items": {k: v.to_json() for k, v in self.items.items()},
        }


class RuleEngine:
    """Turns detections into per-person PPE verdicts. Stateless per frame."""

    def __init__(self, rules: RulesConfig, model: ModelConfig) -> None:
        self.rules = rules
        self.model = model

    # ---------------------------------------------------------------- public

    def evaluate_frame(
        self,
        dets: list[Detection],
        frame_shape: tuple[int, int],
        hints: list[Detection] | None = None,
        frame_bgr=None,
    ) -> list[PersonObservation]:
        """Assess every person in a frame.

        `frame_shape` is (width, height) in pixels. `hints` are weak PPE
        detections (see ModelConfig.hint_conf): one inside a person's box
        turns a MISSING verdict for that item into UNKNOWN, because "the
        model isn't sure it sees a hard hat" is not proof there isn't one.
        With `frame_bgr`, a torso full of hi-vis colour does the same for
        safety-vest (see RulesConfig.hivis_veto_frac).
        """
        persons = [
            d
            for d in dets
            if d.name == PERSON and d.conf >= self.model.conf_for(PERSON)
        ]
        others = [d for d in dets if d.name != PERSON]

        observations = [self._new_observation(p, frame_shape) for p in persons]
        if not observations:
            return []

        self._associate(observations, others, frame_shape)

        for obs in observations:
            for item in self.rules.required_ppe:
                evidence = self._assess_item(obs, item)
                if evidence.is_missing and hints:
                    evidence = self._hint_veto(obs, item, hints) or evidence
                if (
                    evidence.is_missing
                    and item == "safety-vest"
                    and frame_bgr is not None
                    and self.rules.hivis_veto_frac > 0
                ):
                    frac = hivis_fraction(frame_bgr, obs.person.xyxy)
                    if frac >= self.rules.hivis_veto_frac:
                        evidence = ItemEvidence(
                            "UNKNOWN",
                            {"hivis_frac": frac},
                            f"{frac:.0%} of torso is hi-vis colour; vest likely but not detected",
                        )
                if (
                    evidence.is_missing
                    and item == "helmet"
                    and frame_bgr is not None
                    and self.rules.hardhat_colour_veto_frac > 0
                ):
                    heads = obs.attached.get("head") or []
                    if heads:
                        head = max(heads, key=lambda d: d.conf)
                        frac = hardhat_colour_fraction(frame_bgr, head.xyxy)
                        if frac >= self.rules.hardhat_colour_veto_frac:
                            evidence = ItemEvidence(
                                "UNKNOWN",
                                {"hardhat_colour_frac": frac},
                                f"{frac:.0%} of head top is hard-hat yellow; hat likely but not detected",
                            )
                obs.items[item] = evidence
        return observations

    def _hint_veto(
        self, obs: PersonObservation, item: str, hints: list[Detection]
    ) -> ItemEvidence | None:
        """UNKNOWN if a weak detection of this item sits on this person."""
        present_classes = PPE_ITEMS[item][0]
        x1, y1, x2, y2 = obs.person.xyxy
        for hint in hints:
            if hint.name not in present_classes:
                continue
            cx, cy = (hint.x1 + hint.x2) / 2.0, (hint.y1 + hint.y2) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return ItemEvidence(
                    "UNKNOWN",
                    {hint.name: hint.conf},
                    f"possible {hint.name} at {hint.conf:.2f}; too unsure to call it missing",
                )
        return None

    # --------------------------------------------------------------- helpers

    def _new_observation(
        self, person: Detection, frame_shape: tuple[int, int]
    ) -> PersonObservation:
        frame_w, frame_h = frame_shape
        height_frac = person.height / max(1, frame_h)
        truncated = touches_edge(person.xyxy, frame_shape, self.rules.edge_margin_px)
        area_frac = person.area() / max(1, frame_w * frame_h)

        # "Can we see enough of this person to judge them on absence of PPE?"
        # Deliberately conservative: too small, or clipped by the frame, and we
        # decline to use absence evidence at all.
        view_ok = (
            height_frac >= self.rules.min_person_height_frac
            and area_frac >= self.rules.min_person_area_frac
            and not truncated
        )
        return PersonObservation(
            person=person,
            view_ok=view_ok,
            truncated=truncated,
            height_frac=height_frac,
        )

    def _associate(
        self,
        observations: list[PersonObservation],
        others: list[Detection],
        frame_shape: tuple[int, int],
    ) -> None:
        """Attach each non-person detection to at most one person.

        Containment, not IoU (see ppe_types.overlap_frac). Two guards against
        nonsense:

        Ambiguity -- if the best two candidate owners are within
        `assoc_ambiguity_margin` of each other, the item is attached to NOBODY
        and recorded as ambiguous. In a crowd, refusing to answer is correct;
        guessing invents violations.

        Anatomy -- the item must sit in a plausible vertical band of the person
        box. A `helmet` at knee height is a misdetection.
        """
        padded = {
            id(obs): pad_box(obs.person.xyxy, self.rules.person_box_pad, frame_shape)
            for obs in observations
        }

        for det in others:
            if det.conf < self.model.conf_for(det.name):
                continue

            scored: list[tuple[float, PersonObservation]] = []
            for obs in observations:
                frac = overlap_frac(det.xyxy, padded[id(obs)])
                if frac >= self.rules.assoc_containment_min and self._anatomy_ok(
                    det, obs
                ):
                    scored.append((frac, obs))

            if not scored:
                continue

            scored.sort(key=lambda t: t[0], reverse=True)
            if (
                len(scored) > 1
                and scored[0][0] - scored[1][0] < self.rules.assoc_ambiguity_margin
            ):
                # Ambiguous ownership: mark it on every plausible owner so
                # _assess_item can return UNKNOWN rather than a false verdict.
                for _, obs in scored:
                    obs.attached.setdefault("__ambiguous__", []).append(det)
                continue

            scored[0][1].attached.setdefault(det.name, []).append(det)

    def _anatomy_ok(self, det: Detection, obs: PersonObservation) -> bool:
        lo, hi = self.rules.region_gate(det.name)
        if (lo, hi) == (0.0, 1.0):
            return True
        top, bottom = vertical_band(det.xyxy, obs.person.xyxy)
        # Judge by the box's midpoint, with a little slack, so a helmet poking
        # above the person box isn't rejected for being at a negative fraction.
        mid = (top + bottom) / 2.0
        return (lo - 0.10) <= mid <= (hi + 0.10)

    def _best(self, obs: PersonObservation, names: tuple[str, ...]) -> Detection | None:
        best: Detection | None = None
        for name in names:
            for det in obs.attached.get(name, []):
                if best is None or det.conf > best.conf:
                    best = det
        return best

    def _assess_item(self, obs: PersonObservation, item: str) -> ItemEvidence:
        satisfied_by, proven_missing_by = PPE_ITEMS[item]

        ambiguous = obs.attached.get("__ambiguous__", [])
        if any(a.name in satisfied_by or a.name in proven_missing_by for a in ambiguous):
            return ItemEvidence(
                "UNKNOWN",
                {},
                "overlapping people: cannot tell whose equipment this is",
            )

        worn = self._best(obs, satisfied_by)
        worn_vetoes = worn is not None and worn.conf >= self.model.veto_threshold(
            worn.name
        )
        bare = self._best(obs, proven_missing_by) if proven_missing_by else None

        # (1) Both a PPE box and a confident uncovered body part. The PPE must
        #     actually be ON the body part to count -- a helmet carried under
        #     an arm, or sitting on a bench behind someone, lands inside the
        #     person box but is not being worn. Checked before the presence
        #     veto, or the veto would clear them regardless.
        if bare is not None and bare.conf >= self.model.evidence_threshold(bare.name):
            covered = self._covering_ppe(obs, bare, satisfied_by)
            if covered is not None:
                return ItemEvidence(
                    "PRESENT",
                    {bare.name: bare.conf, covered.name: covered.conf},
                    f"{bare.name} covered by {covered.name} "
                    f"({covered.conf:.2f})",
                )
            if worn is not None:
                return ItemEvidence(
                    "MISSING_POSITIVE",
                    {bare.name: bare.conf, worn.name: worn.conf, PERSON: obs.person.conf},
                    f"bare {bare.name} at {bare.conf:.2f}; {worn.name} detected "
                    f"but not covering it (not worn)",
                )
            return ItemEvidence(
                "MISSING_POSITIVE",
                {bare.name: bare.conf, PERSON: obs.person.conf},
                f"bare {bare.name} at {bare.conf:.2f}, no "
                f"{'/'.join(satisfied_by)} detected",
            )

        # (2) Presence veto. PPE seen, and no confident evidence of a bare
        #     body part to contradict it.
        if worn_vetoes:
            return ItemEvidence(
                "PRESENT",
                {worn.name: worn.conf},
                f"{worn.name} detected at {worn.conf:.2f}",
            )

        # (3) A body part we can see but not confidently enough to accuse on.
        if bare is not None:
            threshold = self.model.evidence_threshold(bare.name)
            return ItemEvidence(
                "UNKNOWN",
                {bare.name: bare.conf},
                f"{bare.name} at {bare.conf:.2f} below evidence bar {threshold:.2f}",
            )

        # (4) Positive-evidence item with neither PPE nor body part visible.
        #     We genuinely do not know, and this is NOT a violation. Returning
        #     MISSING here is the single easiest way to make the whole system
        #     cry wolf.
        if proven_missing_by:
            return ItemEvidence(
                "UNKNOWN",
                {},
                f"no {'/'.join(satisfied_by)} and no "
                f"{'/'.join(proven_missing_by)} visible",
            )

        # (3) Absence-only item (safety-vest). Weakest grade of evidence, so
        #     it requires a clear view of the person.
        if self.rules.absence_requires_full_view and not obs.view_ok:
            why = "clipped by frame edge" if obs.truncated else "person too small"
            return ItemEvidence(
                "UNKNOWN", {PERSON: obs.person.conf}, f"view inadequate: {why}"
            )
        return ItemEvidence(
            "MISSING_ABSENT",
            {PERSON: obs.person.conf},
            f"no {'/'.join(satisfied_by)} detected on a clearly-visible person",
        )

    def _covering_ppe(
        self,
        obs: PersonObservation,
        bare: Detection,
        satisfied_by: tuple[str, ...],
    ) -> Detection | None:
        """A PPE box overlapping the bare part enough to count as covering it.

        Guards the case where the detector fires both `head` and `helmet` for
        a helmeted person -- common, and it must read as compliant. Note this
        relies on NMS being class-wise: agnostic NMS would have suppressed one
        of the two boxes and destroyed the comparison.
        """
        for name in satisfied_by:
            for ppe in obs.attached.get(name, []):
                if overlap_frac(bare.xyxy, ppe.xyxy) > self.rules.ppe_cover_overlap_min:
                    return ppe
        return None


def hivis_fraction(frame_bgr, person_xyxy) -> float:
    """Share of the torso region that is fluorescent yellow/green.

    Hi-vis fabric is saturated and bright at hue ~20-45 (OpenCV's 0-180
    scale), which very little else in a room is.
    """
    import cv2

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = person_xyxy
    height = y2 - y1
    ty1, ty2 = int(y1 + 0.20 * height), int(y1 + 0.65 * height)
    tx1, tx2 = int(x1), int(x2)
    ty1, ty2 = max(0, ty1), min(h_img, ty2)
    tx1, tx2 = max(0, tx1), min(w_img, tx2)
    if ty2 - ty1 < 4 or tx2 - tx1 < 4:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (hue >= 20) & (hue <= 45) & (sat >= 90) & (val >= 120)
    return float(mask.mean())


def hardhat_colour_fraction(frame_bgr, head_xyxy) -> float:
    """Share of the top of a head (and just above it) in hard-hat yellow."""
    import cv2

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = head_xyxy
    height = y2 - y1
    ty1, ty2 = int(y1 - 0.15 * height), int(y1 + 0.45 * height)
    ty1, ty2 = max(0, ty1), min(h_img, ty2)
    tx1, tx2 = max(0, int(x1)), min(w_img, int(x2))
    if ty2 - ty1 < 4 or tx2 - tx1 < 4:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (hue >= 18) & (hue <= 40) & (sat >= 110) & (val >= 130)
    return float(mask.mean())


def summarize(observations: list[PersonObservation]) -> str:
    """One-line-per-person summary, for CLI output and patrol logs."""
    if not observations:
        return "no people detected"
    lines = []
    for i, obs in enumerate(observations):
        bits = [
            f"{name}={ev.status}" for name, ev in obs.items.items()
        ]
        flags = []
        if obs.truncated:
            flags.append("truncated")
        if not obs.view_ok:
            flags.append("view-poor")
        lines.append(
            f"  person[{i}] conf={obs.person.conf:.2f} h={obs.height_frac:.2f} "
            f"{' '.join(bits)}{' (' + ', '.join(flags) + ')' if flags else ''}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    from ppe_config import load_config

    ap = argparse.ArgumentParser(description="PPE rule engine.")
    ap.add_argument("--image", help="run the detector on an image, then the rules")
    ap.add_argument("--config", help="config path")
    ap.add_argument("--json", action="store_true", help="dump verdicts as JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = RuleEngine(cfg.rules, cfg.model)

    if not args.image:
        raise SystemExit(
            "nothing to do: pass --image, or run test_ppe_rules.py for the test suite"
        )

    import cv2  # imported here so the module itself stays dependency-free

    from ppe_detector import PpeDetector, annotate

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    det = PpeDetector.from_config(cfg, which="confirm")
    detections = det.detect(frame)
    h, w = frame.shape[:2]
    observations = engine.evaluate_frame(detections, (w, h))

    print(f"{args.image}: {len(detections)} detections, {len(observations)} people")
    for d in detections:
        print(f"  {d.name:14s} {d.conf:.3f} {[round(v) for v in d.xyxy]}")
    print(summarize(observations))
    if args.json:
        print(json.dumps([o.to_json() for o in observations], indent=2))

    out = "rules_annotated.jpg"
    cv2.imwrite(out, annotate(frame, detections, observations))
    print(f"wrote {out}")

"""Single-person association and repeated, settled-frame PPE verdicts."""
from collections import deque
from ppe_types import iou


class InspectionTarget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.box = None
        self.last_seen = 0.0
        self.history = {}

    def update(self, observations, now, recording):
        candidates = [o for o in observations
                      if o.person.conf >= self.cfg.patrol.person_trigger_conf]
        if self.box is None:
            target = max(candidates, key=lambda o: o.person.area(), default=None)
        else:
            target = max(candidates, key=lambda o: iou(self.box, o.person.xyxy), default=None)
            if target is not None and iou(self.box, target.person.xyxy) < 0.2:
                target = None
        if target is None:
            return None, 'UNKNOWN'
        self.box = target.person.xyxy
        self.last_seen = now
        if recording:
            for item in self.cfg.rules.required_ppe:
                evidence = target.items.get(item)
                self.history.setdefault(item, deque(maxlen=self.cfg.temporal.window_m)).append(
                    evidence.status if evidence else 'UNKNOWN')
        verdicts = []
        for item in self.cfg.rules.required_ppe:
            history = self.history.get(item, ())
            n = self.cfg.temporal.positive_n
            if len(history) < self.cfg.temporal.min_track_hits:
                verdicts.append('UNKNOWN')
            elif history.count('PRESENT') >= n and not any(s.startswith('MISSING') for s in history):
                verdicts.append('PRESENT')
            elif history.count('MISSING_POSITIVE') >= n and 'PRESENT' not in history:
                verdicts.append('MISSING')
            elif history.count('MISSING_ABSENT') >= self.cfg.temporal.absence_n and 'PRESENT' not in history:
                verdicts.append('MISSING')
            else:
                verdicts.append('UNKNOWN')
        verdict = ('MISSING' if 'MISSING' in verdicts else
                   'PRESENT' if verdicts and all(v == 'PRESENT' for v in verdicts) else 'UNKNOWN')
        return target, verdict

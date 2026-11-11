import unittest
from ppe_config import load_config
from ppe_inspection import InspectionTarget
from ppe_rules import PersonObservation, ItemEvidence
from ppe_types import Detection


def observation(status, box=(100, 100, 200, 400)):
    return PersonObservation(Detection(0, 'person', .9, box),
                             items={'helmet': ItemEvidence(status)})


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.target = InspectionTarget(load_config())

    def test_unknown_never_becomes_missing_from_elapsed_time(self):
        for i in range(100):
            _, verdict = self.target.update([observation('UNKNOWN')], i, True)
            self.assertEqual(verdict, 'UNKNOWN')

    def test_present_and_missing_need_repeated_settled_frames(self):
        for status, result in [('PRESENT', 'PRESENT'), ('MISSING_POSITIVE', 'MISSING')]:
            target = InspectionTarget(load_config())
            for i in range(20):
                self.assertEqual(target.update([observation(status)], i, False)[1], 'UNKNOWN')
            for i in range(4):
                self.assertEqual(target.update([observation(status)], i, True)[1], 'UNKNOWN')
            self.assertEqual(target.update([observation(status)], 5, True)[1], result)

    def test_another_person_cannot_supply_target_evidence(self):
        self.target.update([observation('UNKNOWN')], 0, True)
        for i in range(10):
            target, verdict = self.target.update([observation('PRESENT', (400, 100, 500, 400))], i+1, True)
            self.assertIsNone(target)
            self.assertEqual(verdict, 'UNKNOWN')

    def test_conflicting_evidence_stays_unknown(self):
        for i in range(10):
            _, verdict = self.target.update([observation('PRESENT' if i % 2 else 'MISSING_POSITIVE')], i, True)
        self.assertEqual(verdict, 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()

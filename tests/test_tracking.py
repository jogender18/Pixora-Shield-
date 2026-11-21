import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.app import RegionTracker


class RegionTrackerTest(unittest.TestCase):
    def test_keeps_second_face_track_when_one_detection_is_missing(self):
        tracker = RegionTracker(max_missed=10, match_threshold=0.12, smooth=0.45)
        first = tracker.update([[10, 10, 80, 80], [300, 20, 370, 90]])
        second = tracker.update([[12, 12, 82, 82]])

        self.assertEqual(len(first), 2)
        self.assertGreaterEqual(len(second), 1)
        self.assertTrue(any(box[0] >= 290 for box in second))


if __name__ == "__main__":
    unittest.main()

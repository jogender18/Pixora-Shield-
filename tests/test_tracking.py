import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.app import RegionTracker, is_visible_face, overlap_ratio


class RegionTrackerTest(unittest.TestCase):
    def test_visible_face_filter_rejects_partial_or_tiny_faces(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.assertTrue(is_visible_face(frame, [500, 150, 650, 300]))
        self.assertFalse(is_visible_face(frame, [10, 10, 30, 30]))
        self.assertFalse(is_visible_face(frame, [500, 150, 700, 210]))
    def test_excludes_overlapping_main_face_from_blur_tracks(self):
        main_face = [100, 100, 200, 220]
        other_face = [300, 110, 390, 220]
        detections = [
            box for box in [main_face, other_face]
            if overlap_ratio(main_face, box) < 0.40
        ]

        self.assertEqual(detections, [other_face])
    def test_keeps_second_face_track_when_one_detection_is_missing(self):
        tracker = RegionTracker(max_missed=10, match_threshold=0.12, smooth=0.45)
        first = tracker.update([[10, 10, 80, 80], [300, 20, 370, 90]])
        second = tracker.update([[12, 12, 82, 82]])

        self.assertEqual(len(first), 2)
        self.assertGreaterEqual(len(second), 1)
        self.assertTrue(any(box[0] >= 290 for box in second))


if __name__ == "__main__":
    unittest.main()

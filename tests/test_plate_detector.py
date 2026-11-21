import unittest
import cv2
import numpy as np

from backend.src.models.detection.plate_blur.plate_detector import PlateDetector


class PlateDetectorTest(unittest.TestCase):
    def test_fallback_detects_rectangular_plate_region(self):
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (220, 300), (420, 350), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "ABC123",
            (235, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            2,
        )

        _, regions = PlateDetector().process_frame(frame, 0)

        self.assertTrue(
            any(
                region[0] <= 220
                and region[1] <= 300
                and region[2] >= 420
                and region[3] >= 350
                for region in regions
            )
        )

    def test_plate_region_stays_blurred_when_detection_misses_one_frame(self):
        detector = PlateDetector()
        plate_frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(plate_frame, (220, 300), (420, 350), (20, 20, 20), -1)
        cv2.putText(
            plate_frame,
            "ABC123",
            (235, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            2,
        )

        _, first_regions = detector.process_frame(plate_frame, 0)
        _, missed_regions = detector.process_frame(
            np.full((480, 640, 3), 220, dtype=np.uint8),
            1,
        )

        self.assertTrue(first_regions)
        self.assertEqual(missed_regions, first_regions)


if __name__ == "__main__":
    unittest.main()

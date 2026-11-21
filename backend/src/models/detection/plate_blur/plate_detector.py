import os
from pathlib import Path
import cv2


class PlateDetector:
    def __init__(self, weights_path=None, conf_thresh=0.25):
        self.weights_path = weights_path or os.getenv("PLATE_MODEL_PATH")
        self.conf_thresh = conf_thresh
        self.model = None
        if self.weights_path and Path(self.weights_path).is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception:
                self.model = None

    def fallback_regions(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        frame_area = height * width
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 55 or box_height < 18:
                continue

            aspect = box_width / float(box_height)
            area = box_width * box_height
            if not 1.8 <= aspect <= 6.5:
                continue
            if not 0.0004 * frame_area <= area <= 0.14 * frame_area:
                continue

            contour_area = cv2.contourArea(contour)
            if contour_area / max(area, 1) < 0.18:
                continue

            padding_x = int(box_width * 0.06)
            padding_y = int(box_height * 0.12)
            regions.append([
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(width, x + box_width + padding_x),
                min(height, y + box_height + padding_y),
            ])

        regions.sort(key=lambda region: (
            (region[2] - region[0]) * (region[3] - region[1])
        ), reverse=True)
        return regions[:8]

    def process_frame(self, frame, frame_id):
        if self.model is None:
            return frame_id, self.fallback_regions(frame)

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False,
        )
        rectangles = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                rectangles.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ])
        return frame_id, rectangles

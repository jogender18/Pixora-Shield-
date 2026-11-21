import os
from pathlib import Path
import cv2


class PlateDetector:
    def __init__(self, weights_path=None, conf_thresh=0.25):
        self.weights_path = weights_path or os.getenv("PLATE_MODEL_PATH")
        self.conf_thresh = conf_thresh
        self.model = None
        self.previous_regions = []
        self.missed_frames = 0
        self.max_missed_frames = 3
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
            if box_width < 45 or box_height < 14:
                continue

            aspect = box_width / float(box_height)
            area = box_width * box_height
            if not 2.0 <= aspect <= 7.5:
                continue
            if not 0.0003 * frame_area <= area <= 0.12 * frame_area:
                continue

            contour_area = cv2.contourArea(contour)
            if contour_area / max(area, 1) < 0.12:
                continue

            roi = gray[y:y + box_height, x:x + box_width]
            if roi.size == 0:
                continue

            roi_std = float(cv2.meanStdDev(roi)[1][0][0])
            if roi_std < 8:
                continue

            padding_x = int(box_width * 0.03)
            padding_y = int(box_height * 0.08)
            regions.append([
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(width, x + box_width + padding_x),
                min(height, y + box_height + padding_y),
            ])

        regions.sort(key=lambda region: (
            (region[2] - region[0]) * (region[3] - region[1])
        ), reverse=True)
        filtered = []
        for region in regions:
            duplicate = False
            x1, y1, x2, y2 = region
            for kept in filtered:
                px1, py1, px2, py2 = kept
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = max(1, (x2 - x1) * (y2 - y1))
                kept_area = max(1, (px2 - px1) * (py2 - py1))
                iou = intersection / (area + kept_area - intersection)
                if iou > 0.25:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(region)
        return filtered[:2]

    def reset(self):
        self.previous_regions = []
        self.missed_frames = 0

    def process_frame(self, frame, frame_id):
        if self.model is None:
            regions = self.fallback_regions(frame)
        else:
            regions = self._yolo_regions(frame)

        if regions:
            matched, unmatched_previous = self._match_regions(regions)
            regions = matched
            self.previous_regions = regions
            self.missed_frames = 0
        elif self.previous_regions and self.missed_frames < self.max_missed_frames:
            regions = self.previous_regions
            self.missed_frames += 1
        else:
            regions = []
            self.previous_regions = []

        return frame_id, regions

    def _yolo_regions(self, frame):
        if self.model is None:
            return []

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
        return rectangles

    def _match_regions(self, regions):
        if not self.previous_regions:
            return regions, []

        matched = []
        unmatched_previous = list(self.previous_regions)
        for region in regions:
            best_previous = None
            best_iou = 0.45
            x1, y1, x2, y2 = region
            for previous in self.previous_regions:
                px1, py1, px2, py2 = previous
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = max(1, (x2 - x1) * (y2 - y1))
                previous_area = max(1, (px2 - px1) * (py2 - py1))
                iou = intersection / (area + previous_area - intersection)
                if iou >= best_iou:
                    best_iou = iou
                    best_previous = previous

            if best_previous is not None:
                nx1 = int((x1 + best_previous[0]) // 2)
                ny1 = int((y1 + best_previous[1]) // 2)
                nx2 = int((x2 + best_previous[2]) // 2)
                ny2 = int((y2 + best_previous[3]) // 2)
                matched.append([nx1, ny1, nx2, ny2])
                if best_previous in unmatched_previous:
                    unmatched_previous.remove(best_previous)
            else:
                matched.append(region)

        return matched, unmatched_previous

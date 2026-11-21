import os
import re
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, send_from_directory
from flask import request
from flask_cors import CORS

from src.models.detection.blur_utils import apply_blur_regions
from src.models.detection.plate_blur.plate_detector import PlateDetector


BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","))

def find_face_cascade():
    candidates = []
    candidates.append(
        str(BACKEND_DIR / "models" / "haarcascade_frontalface_default.xml")
    )
    if hasattr(cv2, "data"):
        candidates.append(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
    candidates.extend([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ])
    return next((path for path in candidates if Path(path).is_file()), "")


face_cascade_path = find_face_cascade()
try:
    face_detector = (
        cv2.CascadeClassifier(face_cascade_path)
        if face_cascade_path and hasattr(cv2, "CascadeClassifier")
        else None
    )
except cv2.error:
    face_detector = None
plate_detector = PlateDetector(conf_thresh=float(os.getenv("PLATE_CONF_THRESH", "0.25")))
try:
    import pytesseract
except ImportError:
    pytesseract = None


PII_PATTERNS = (
    re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)


def detect_faces(frame):
    if face_detector is None or face_detector.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    return [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in faces]


def detect_pii_regions(frame, frame_id):
    if pytesseract is None or frame_id % 30 != 0:
        return []

    try:
        data = pytesseract.image_to_data(
            frame,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    regions = []
    for index, text in enumerate(data.get("text", [])):
        text = text.strip()
        if not text or not any(pattern.search(text) for pattern in PII_PATTERNS):
            continue
        x = int(data["left"][index])
        y = int(data["top"][index])
        w = int(data["width"][index])
        h = int(data["height"][index])
        regions.append([x, y, x + w, y + h])
    return regions


def stabilize_regions(previous_regions, current_regions, iou_threshold=0.55):
    if not previous_regions:
        return current_regions

    stabilized = []
    for region in current_regions:
        match = None
        best_iou = iou_threshold
        x1, y1, x2, y2 = region
        for previous in previous_regions:
            px1, py1, px2, py2 = previous
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = max(1, (x2 - x1) * (y2 - y1))
            previous_area = max(1, (px2 - px1) * (py2 - py1))
            iou = intersection / (area + previous_area - intersection)
            if iou >= best_iou:
                best_iou = iou
                match = previous
        stabilized.append(match if match is not None else region)
    return stabilized


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "face_detection": bool(face_detector is not None and not face_detector.empty()),
        "plate_detection": plate_detector.model is not None,
    })


@app.post("/api/process")
def process_video():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "A video file is required"}), 400

    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return jsonify({"error": "Unsupported video format"}), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    output_name = f"{job_id}_protected.mp4"
    output_path = OUTPUT_DIR / output_name
    video.save(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not open the uploaded video"}), 400

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not create the output video"}), 500

    started = time.perf_counter()
    processed_frames = 0
    detected_regions = 0
    frame_id = 0
    previous_face_regions = []
    pii_regions_detected = 0
    plate_regions_detected = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        face_regions = detect_faces(frame)
        face_regions = stabilize_regions(previous_face_regions, face_regions)
        previous_face_regions = face_regions
        _, plate_regions = plate_detector.process_frame(frame, frame_id)
        pii_regions = detect_pii_regions(frame, frame_id)
        regions = face_regions + plate_regions + pii_regions
        detected_regions += len(regions)
        plate_regions_detected += len(plate_regions)
        pii_regions_detected += len(pii_regions)
        processed = apply_blur_regions(frame, regions)
        writer.write(processed)
        processed_frames += 1
        frame_id += 1

    capture.release()
    writer.release()

    elapsed = time.perf_counter() - started
    original_url = f"/api/original/{input_path.name}"
    processed_url = f"/api/output/{output_name}"
    return jsonify({
        "job_id": job_id,
        "original_url": original_url,
        "processed_url": processed_url,
        "pipeline_summary": (
            f"Processed {processed_frames} frames: "
            f"{detected_regions} privacy regions, "
            f"{plate_regions_detected} plate detections, "
            f"and {pii_regions_detected} PII text detections."
        ),
        "metrics": {
            "frames": processed_frames,
            "reported_frames": total_frames,
            "faces_and_plates_detected": detected_regions,
            "processing_seconds": round(elapsed, 2),
            "average_fps": round(processed_frames / elapsed, 2) if elapsed else 0,
            "resolution": f"{width}x{height}",
            "plate_detection_enabled": plate_detector.model is not None,
        },
    })


@app.get("/api/original/<path:filename>")
def serve_original(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/api/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/api/download/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

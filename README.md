# Pixora Shield

Pixora Shield uploads a video, detects faces and optional license plates, and
produces an MP4 with those regions blurred.

## Project structure

- `frontend/index.html` - Page markup
- `frontend/styles.css` - All visual styles
- `frontend/app.js` - Upload, API, drag-and-drop, and result behavior
- `backend/app.py` - Flask API and video processing endpoints
- `backend/run.py` - Backend server entry point
- `backend/requirements.txt` - Python dependencies
- `backend/models/` - Detection model files

## Run locally

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.run
```

Frontend:

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000`. The API runs at `http://localhost:5000`.



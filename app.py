"""
AgroVision FastAPI backend: single-image crop disease prediction.
Run: uvicorn app:app --reload
"""
from __future__ import annotations

import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from inference.predict import load_model, predict_image

# Paths relative to project root (cwd when running uvicorn)
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
DISEASE_INFO_PATH = PROJECT_ROOT / "inference" / "disease_info.json"

_model: nn.Module | None = None
_class_names: list[str] | None = None
_device: torch.device | None = None
_disease_info: Dict[str, Dict[str, str]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global _model, _class_names, _device, _disease_info
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model_path = str(DEFAULT_MODEL_PATH)
        _model, _class_names, _device = load_model(model_path=model_path)
    except FileNotFoundError as e:
        _model, _class_names, _device = None, None, None
        print(f"[startup] Model not loaded: {e}")
    except Exception as e:
        _model, _class_names, _device = None, None, None
        print(f"[startup] Model load failed: {e}")
    try:
        if DISEASE_INFO_PATH.exists():
            with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                _disease_info = raw
            else:
                _disease_info = {}
        else:
            _disease_info = {}
    except Exception as e:
        _disease_info = {}
        print(f"[startup] Failed to load disease info: {e}")
    yield


app = FastAPI(title="AgroVision API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/")
def root(request: Request) -> HTMLResponse:
    """Serve the frontend web interface."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept image upload; return predicted class and confidence."""
    if _model is None or _class_names is None or _device is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train and save models/best_model.pth or check server logs.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {suffix}. Allowed: {sorted(allowed)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / safe_name

    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save upload: {e}") from e
    finally:
        await file.close()

    try:
        result = predict_image(
            str(dest),
            model_path=str(DEFAULT_MODEL_PATH),
            model=_model,
            class_names=_class_names,
            device=_device,
        )
    except ValueError as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        if dest.exists():
            dest.unlink(missing_ok=True)

    return {
        "predicted_class": result["class"],
        "confidence": result["confidence"],
        "description": _disease_info.get(result["class"], {}).get(
            "description",
            "No disease description is available yet for this class.",
        ),
        "treatment": _disease_info.get(result["class"], {}).get(
            "treatment",
            "Treatment guidance is not available yet. Consult a local agronomist.",
        ),
        "prevention": _disease_info.get(result["class"], {}).get(
            "prevention",
            "Prevention tips are not available yet. Follow standard crop hygiene practices.",
        ),
    }

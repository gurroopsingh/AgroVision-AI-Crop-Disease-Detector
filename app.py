"""
AgroVision FastAPI backend – Phase 8 production-ready API.

Endpoints:
  GET  /           → serve frontend web UI
  GET  /health     → model status, num_classes, model_name
  GET  /classes    → list of all disease class names
  POST /predict    → single-image prediction
  POST /predict/batch → multi-image prediction

Run: uvicorn app:app --reload
"""
from __future__ import annotations

import io
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

from inference.predict import load_model, predict_image

# ─────────────────────────── Paths ────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent
UPLOAD_DIR         = PROJECT_ROOT / "outputs" / "uploads"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TEMPLATES_DIR      = PROJECT_ROOT / "templates"
STATIC_DIR         = PROJECT_ROOT / "static"
DISEASE_INFO_PATH  = PROJECT_ROOT / "inference" / "disease_info.json"
DISEASE_INFO_HI_PATH = PROJECT_ROOT / "inference" / "disease_info_hi.json"

# ─────────────────────────── Limits ───────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024          # 5 MB
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_MIME_TYPES  = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# ─────────────────────────── Global state ─────────────────────────────────────
_model:       Optional[nn.Module]        = None
_class_names: Optional[List[str]]       = None
_device:      Optional[torch.device]    = None
_disease_info: Dict[str, Dict[str, str]] = {}
_disease_info_hi: Dict[str, Dict[str, str]] = {}

_DEFAULT_DISEASE_INFO: Dict[str, str] = {
    "description": "No disease description is available yet for this class.",
    "treatment":   "Treatment guidance is not available. Consult a local agronomist.",
    "prevention":  "Follow standard crop hygiene and pest management practices.",
    "fertilizer":  "Maintain balanced NPK fertilization based on soil test results. "
                   "Consult your local agricultural extension officer for crop-specific recommendations.",
}

_HEALTHY_INFO: Dict[str, str] = {
    "description": "The plant appears healthy with no visible signs of disease.",
    "treatment":   "No treatment required. Continue regular monitoring.",
    "prevention":  "Maintain good agricultural practices: proper spacing, irrigation, and sanitation.",
    "fertilizer":  "Apply balanced NPK fertilizer as per crop schedule and soil test. "
                   "Organic compost is beneficial for long-term soil health.",
}


# ─────────────────────────── Lifespan ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and disease info once at startup."""
    global _model, _class_names, _device, _disease_info, _disease_info_hi

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        _model, _class_names, _device = load_model(model_path=str(DEFAULT_MODEL_PATH))
        print(f"[startup] Model loaded: {len(_class_names)} classes on {_device}")
    except FileNotFoundError as e:
        print(f"[startup] Model not loaded (FileNotFoundError): {e}")
    except Exception as e:
        print(f"[startup] Model load failed: {e}")

    # Load disease info (English)
    try:
        if DISEASE_INFO_PATH.exists():
            with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _disease_info = raw if isinstance(raw, dict) else {}
        else:
            _disease_info = {}
    except Exception as e:
        _disease_info = {}
        print(f"[startup] Failed to load disease_info.json: {e}")

    # Load disease info (Hindi)
    try:
        if DISEASE_INFO_HI_PATH.exists():
            with open(DISEASE_INFO_HI_PATH, "r", encoding="utf-8") as f:
                raw_hi = json.load(f)
            _disease_info_hi = raw_hi if isinstance(raw_hi, dict) else {}
        else:
            _disease_info_hi = {}
    except Exception as e:
        _disease_info_hi = {}
        print(f"[startup] Failed to load disease_info_hi.json: {e}")

    yield


# ─────────────────────────── App ──────────────────────────────────────────────
app = FastAPI(
    title="AgroVision API",
    description="AI-powered crop disease detection backend.",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────── Helpers ──────────────────────────────────────────
def _get_disease_entry(class_name: str, lang: str = "en") -> Dict[str, str]:
    """Return disease info dict for a class; fall back to defaults."""
    source_dict = _disease_info_hi if lang == "hi" else _disease_info
    if class_name.startswith("healthy_"):
        entry = _HEALTHY_INFO.copy()
        entry.update(source_dict.get(class_name, {}))
        return entry
    entry = _DEFAULT_DISEASE_INFO.copy()
    entry.update(source_dict.get(class_name, {}))
    return entry


def _model_name_display() -> str:
    """Human-readable model name from checkpoint or fallback."""
    return "ResNet18 Transfer Learning"


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    """
    Read upload bytes with size + extension validation.
    Returns raw bytes. Raises HTTPException on any problem.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    raw = await file.read()
    await file.close()

    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) // 1024} KB). Maximum allowed size is 5 MB.",
        )

    # Verify it is actually a readable image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid or readable image.")

    return raw


def _correct_exif_orientation(raw: bytes) -> bytes:
    """Apply EXIF rotation so portrait phone photos are upright before inference."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img)
            buf = io.BytesIO()
            fmt = img.format or "JPEG"
            img.save(buf, format=fmt)
            return buf.getvalue()
    except Exception:
        return raw  # if anything goes wrong, pass through unchanged


def _save_temp(raw: bytes, suffix: str) -> Path:
    """Save bytes to a temporary file in UPLOAD_DIR; return path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(raw)
    return dest


# ─────────────────────────── Routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """Serve the legacy HTML frontend (kept for backward compatibility)."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Return API and model health status.

    Response:
        status       : "ok" or "degraded"
        model_loaded : bool
        num_classes  : int (0 if model not loaded)
        model_name   : str
    """
    loaded = _model is not None and _class_names is not None
    return {
        "status":       "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "num_classes":  len(_class_names) if _class_names else 0,
        "model_name":   _model_name_display(),
    }


@app.get("/classes")
def get_classes() -> Dict[str, Any]:
    """
    Return the ordered list of all disease class names.

    Response:
        classes    : List[str]
        num_classes: int
    """
    if _class_names is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Class names unavailable.",
        )
    return {
        "classes":     _class_names,
        "num_classes": len(_class_names),
    }


@app.get("/disease-info")
def get_disease_info(lang: str = "en") -> Dict[str, Any]:
    """
    Return the full disease info dictionary and class names for the frontend Encyclopedia.
    """
    source_dict = _disease_info_hi if lang == "hi" else _disease_info
    if not source_dict or not _class_names:
        raise HTTPException(
            status_code=503,
            detail="Disease info or class names not loaded.",
        )
    return {
        "classes": _class_names,
        "disease_info": source_dict
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en") -> Dict[str, Any]:
    """
    Accept a single image upload; return predicted class, confidence, and agronomic info.

    Response:
        predicted_class : str
        confidence      : float  (0.0 – 1.0)
        description     : str
        treatment       : str
        prevention      : str
        fertilizer      : str
    """
    if _model is None or _class_names is None or _device is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure models/best_model.pth exists.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    raw    = await _read_and_validate_upload(file)
    raw    = _correct_exif_orientation(raw)
    dest   = _save_temp(raw, suffix)

    try:
        result = predict_image(
            str(dest),
            model_path=str(DEFAULT_MODEL_PATH),
            model=_model,
            class_names=_class_names,
            device=_device,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        dest.unlink(missing_ok=True)

    info = _get_disease_entry(result["class"], lang=lang)
    return {
        "predicted_class": result["class"],
        "confidence":      result["confidence"],
        "description":     info["description"],
        "treatment":       info["treatment"],
        "prevention":      info["prevention"],
        "fertilizer":      info["fertilizer"],
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), lang: str = "en") -> Dict[str, Any]:
    """
    Accept multiple image uploads; return predictions for each.

    Response:
        predictions: List[{filename, predicted_class, confidence, description,
                            treatment, prevention, fertilizer}]
        total      : int
    """
    if _model is None or _class_names is None or _device is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure models/best_model.pth exists.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch request.")

    predictions: List[Dict[str, Any]] = []
    temp_paths: List[Path] = []

    try:
        for upload in files:
            suffix   = Path(upload.filename or "img").suffix.lower()
            raw      = await _read_and_validate_upload(upload)
            raw      = _correct_exif_orientation(raw)
            dest     = _save_temp(raw, suffix)
            temp_paths.append(dest)

            try:
                result = predict_image(
                    str(dest),
                    model_path=str(DEFAULT_MODEL_PATH),
                    model=_model,
                    class_names=_class_names,
                    device=_device,
                )
                info = _get_disease_entry(result["class"], lang=lang)
                predictions.append({
                    "filename":       upload.filename or "unknown",
                    "predicted_class": result["class"],
                    "confidence":     result["confidence"],
                    "description":    info["description"],
                    "treatment":      info["treatment"],
                    "prevention":     info["prevention"],
                    "fertilizer":     info["fertilizer"],
                })
            except (ValueError, FileNotFoundError) as e:
                predictions.append({
                    "filename": upload.filename or "unknown",
                    "error":    str(e),
                })
    finally:
        for p in temp_paths:
            p.unlink(missing_ok=True)

    return {
        "predictions": predictions,
        "total":       len(predictions),
    }

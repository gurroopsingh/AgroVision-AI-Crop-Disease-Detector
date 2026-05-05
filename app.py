"""
AgroVision FastAPI backend – Phase 9: Gemini-powered disease insights.

Endpoints:
  GET  /health         → model status, num_classes, model_name
  GET  /classes        → list of all disease class names
  GET  /disease-info   → full JSON disease info (Encyclopedia – unchanged)
  POST /predict        → single-image prediction + Gemini AI description
  POST /predict/batch  → multi-image prediction + Gemini AI descriptions

Run: uvicorn app:app --reload
"""
from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

# Load .env so GEMINI_API_KEY can be set there during local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed – fall back to environment only

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from inference.predict import load_model, predict_image

# ─────────────────────────── Paths ────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent
UPLOAD_DIR         = PROJECT_ROOT / "outputs" / "uploads"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
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
_gemini_client = None  # google-genai Client instance

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


# ─────────────────────────── Gemini helpers ────────────────────────────────────

_GEMINI_PROMPT_EN = """You are a senior agronomist and plant pathologist with 20 years of field experience.
A crop disease AI has detected: **{class_name}** (Confidence: {confidence:.1%})

Respond ONLY with a valid JSON object (no markdown, no code fences) using this EXACT structure:

{{
  "description": "Write 2-3 clear sentences covering: (1) what this disease is and its causal organism (fungus/bacteria/virus/nematode with scientific name), (2) how it spreads and what conditions favour it, (3) the key visual symptoms on leaves/stem/fruit a farmer would see.",

  "treatment": "Write a NUMBERED step-by-step treatment plan. Each step must be on its own line starting with a number and period. ALWAYS include: product name, exact quantity/dose (e.g. 2.5 g per litre of water, 500 mL per acre), method of application, and how many times to apply. Example format:\n1. Spray Mancozeb 75% WP at 2.5 g/L water. Apply 500 L/ha. Repeat every 7 days for 3 applications.\n2. For severe infection, switch to Propiconazole 25% EC at 1 mL/L water. Apply as a drench.\n3. Remove and burn infected plant debris immediately after treatment.\n4. Organic option: Spray Trichoderma viride (10^8 CFU/g) at 5 g/L water every 10 days.",

  "prevention": "Write 4-5 bullet points (each starting with a dash -). Each point should be a specific, actionable preventive measure covering: crop rotation interval, resistant variety names if available, field sanitation steps, optimal plant spacing, humidity/irrigation management, and scouting frequency.",

  "fertilizer": "Write specific fertilizer recommendations in numbered steps. Include: (1) basal dose with NPK ratio and kg/ha quantity, (2) top-dressing schedule with timing and quantities, (3) micronutrients that boost disease resistance (e.g. Zinc sulphate at 25 kg/ha), (4) organic amendments like FYM or compost with quantities per acre, (5) what to AVOID (e.g. excess nitrogen)."
}}

RULES:
- Every quantity MUST have units (g, mL, kg, L, per litre, per acre, per hectare, etc.)
- Use real commercial product names available in India (e.g. Dithane M-45, Bavistin, COC, etc.)
- The treatment field MUST have at least 4 numbered steps
- Do NOT use vague language like 'appropriate amount' or 'as needed'
- Output ONLY the JSON object"""

_GEMINI_PROMPT_HI = """आप 20 साल के अनुभव वाले वरिष्ठ कृषि वैज्ञानिक हैं।
AI ने पहचानी बीमारी: **{class_name}** (विश्वास: {confidence:.1%})

केवल एक valid JSON object दें (कोई markdown नहीं, कोई code fence नहीं):

{{
  "description": "2-3 वाक्यों में बताएं: (1) यह बीमारी क्या है और इसका कारक जीव (scientific नाम सहित), (2) यह कैसे फैलती है, (3) पत्तियों/तने/फल पर दिखने वाले मुख्य लक्षण।",

  "treatment": "क्रमांकित चरण-दर-चरण उपचार योजना लिखें। हर चरण नई लाइन पर हो जिसमें: उत्पाद का नाम, सटीक मात्रा (जैसे 2.5 ग्राम प्रति लीटर पानी, 500 मिली प्रति एकड़), छिड़काव विधि, और कितनी बार करें। कम से कम 4 चरण होने चाहिए।",

  "prevention": "4-5 बुलेट पॉइंट (हर एक - से शुरू): फसल चक्र अंतराल, प्रतिरोधी किस्में, खेत स्वच्छता, पौधों की दूरी, सिंचाई प्रबंधन, और निगरानी आवृत्ति।",

  "fertilizer": "क्रमांकित सिफारिशें: (1) आधार खाद NPK अनुपात और kg/हेक्टेयर मात्रा, (2) ऊपरी खाद समय-सारणी, (3) रोग प्रतिरोधक सूक्ष्म पोषक तत्व (जैसे जिंक सल्फेट 25 kg/ha), (4) FYM/कम्पोस्ट मात्रा प्रति एकड़, (5) क्या न करें।"
}}

नियम: हर मात्रा में इकाई होनी चाहिए। वास्तविक उत्पाद नाम उपयोग करें। केवल JSON दें।"""


def _build_gemini_prompt(class_name: str, confidence: float, lang: str) -> str:
    """Build a Gemini prompt for the given disease class."""
    display_name = class_name.replace("_", " ").title()
    template = _GEMINI_PROMPT_HI if lang == "hi" else _GEMINI_PROMPT_EN
    return template.format(class_name=display_name, confidence=confidence)


async def _get_gemini_disease_info(
    class_name: str, confidence: float, lang: str
) -> Dict[str, str]:
    """
    Query Gemini for detailed disease info using the google-genai SDK.
    Falls back to the JSON-based entry on any error.
    """
    # Short-circuit for healthy plants – Gemini isn't needed
    if class_name.startswith("healthy_"):
        return _get_disease_entry(class_name, lang=lang)

    if not _GEMINI_AVAILABLE or _gemini_client is None:
        print("[gemini] Gemini not available – using JSON fallback")
        return _get_disease_entry(class_name, lang=lang)

    try:
        prompt = _build_gemini_prompt(class_name, confidence, lang)
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text.strip()

        # Strip accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        # Ensure all required keys exist
        required = {"description", "treatment", "prevention", "fertilizer"}
        if not required.issubset(parsed.keys()):
            raise ValueError(f"Gemini response missing keys: {required - set(parsed.keys())}")

        return {
            "description": str(parsed["description"]),
            "treatment":   str(parsed["treatment"]),
            "prevention":  str(parsed["prevention"]),
            "fertilizer":  str(parsed["fertilizer"]),
            "ai_generated": True,
        }

    except Exception as e:
        import traceback
        print(f"[gemini] ERROR for '{class_name}': {type(e).__name__}: {e}")
        traceback.print_exc()
        return _get_disease_entry(class_name, lang=lang)


# ─────────────────────────── Lifespan ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, disease info, and configure Gemini at startup."""
    global _model, _class_names, _device, _disease_info, _disease_info_hi, _gemini_client

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        _model, _class_names, _device = load_model(model_path=str(DEFAULT_MODEL_PATH))
        print(f"[startup] Model loaded: {len(_class_names)} classes on {_device}")
    except FileNotFoundError as e:
        print(f"[startup] Model not loaded (FileNotFoundError): {e}")
    except Exception as e:
        print(f"[startup] Model load failed: {e}")

    # Load disease info (English) – still used for Encyclopedia
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

    # Load disease info (Hindi) – still used for Encyclopedia
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

    # Configure Gemini (new google-genai SDK)
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if _GEMINI_AVAILABLE and gemini_api_key:
        try:
            _gemini_client = google_genai.Client(api_key=gemini_api_key)
            print("[startup] Gemini AI configured successfully (google-genai SDK, gemini-2.0-flash)")
        except Exception as e:
            _gemini_client = None
            print(f"[startup] Gemini configuration failed: {e}")
    else:
        _gemini_client = None
        if not gemini_api_key:
            print("[startup] GEMINI_API_KEY not set – predictions will use JSON fallback")
        else:
            print("[startup] google-genai not installed – using JSON fallback")

    yield


# ─────────────────────────── App ──────────────────────────────────────────────
app = FastAPI(
    title="AgroVision API",
    description="AI-powered crop disease detection backend with Gemini AI insights.",
    version="3.0.0",
    lifespan=lifespan,
)


# ─────────────────────────── Helpers ──────────────────────────────────────────
def _get_disease_entry(class_name: str, lang: str = "en") -> Dict[str, str]:
    """Return disease info dict for a class from JSON; fall back to defaults."""
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


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Return API and model health status.

    Response:
        status           : "ok" or "degraded"
        model_loaded     : bool
        num_classes      : int (0 if model not loaded)
        model_name       : str
        gemini_available : bool
    """
    loaded = _model is not None and _class_names is not None
    return {
        "status":            "ok" if loaded else "degraded",
        "model_loaded":      loaded,
        "num_classes":       len(_class_names) if _class_names else 0,
        "model_name":        _model_name_display(),
        "gemini_available":  _gemini_client is not None,
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
    Return the full disease info dictionary and class names for the Encyclopedia.
    This endpoint continues to use the JSON files (not Gemini).
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
    Accept a single image upload; return predicted class, confidence,
    and Gemini AI-generated agronomic info (falls back to JSON if unavailable).

    Response:
        predicted_class : str
        confidence      : float  (0.0 – 1.0)
        description     : str
        treatment       : str
        prevention      : str
        fertilizer      : str
        ai_generated    : bool   (True if powered by Gemini)
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

    # Use Gemini for rich disease info; falls back to JSON automatically
    info = await _get_gemini_disease_info(result["class"], result["confidence"], lang=lang)

    return {
        "predicted_class": result["class"],
        "confidence":      result["confidence"],
        "description":     info["description"],
        "treatment":       info["treatment"],
        "prevention":      info["prevention"],
        "fertilizer":      info["fertilizer"],
        "ai_generated":    info.get("ai_generated", False),
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), lang: str = "en") -> Dict[str, Any]:
    """
    Accept multiple image uploads; return predictions for each.

    Response:
        predictions: List[{filename, predicted_class, confidence, description,
                            treatment, prevention, fertilizer, ai_generated}]
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
                info = await _get_gemini_disease_info(
                    result["class"], result["confidence"], lang=lang
                )
                predictions.append({
                    "filename":        upload.filename or "unknown",
                    "predicted_class": result["class"],
                    "confidence":      result["confidence"],
                    "description":     info["description"],
                    "treatment":       info["treatment"],
                    "prevention":      info["prevention"],
                    "fertilizer":      info["fertilizer"],
                    "ai_generated":    info.get("ai_generated", False),
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

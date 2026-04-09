"""
Single-image inference using the same validation preprocessing as training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

# Project root on sys.path when running as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import get_model
from training.transforms import get_val_test_transform
from training.utils import load_checkpoint

DEFAULT_MODEL_PATH = "models/best_model.pth"
DEFAULT_LABELS_PATH = "inference/class_labels.json"
DEFAULT_DATASET_TRAIN = "data/MasterDataset/train"

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _project_root() -> Path:
    return _ROOT


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = _project_root() / p
    return p.resolve()


def class_names_from_dataset(train_dir: Path) -> List[str]:
    """Sorted class folder names under train split (same order as DataLoader)."""
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Dataset train directory not found: {train_dir}")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def ensure_class_labels_json(
    labels_path: Path,
    train_dir: Path,
    checkpoint_class_names: List[str] | None = None,
) -> Dict[str, str]:
    """
    Ensure inference/class_labels.json exists with index -> class name mapping.

    Priority: checkpoint class order if provided (writes/updates file), else existing JSON,
    else scan train_dir and create file.
    """
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str]
    if checkpoint_class_names:
        mapping = {str(i): name for i, name in enumerate(checkpoint_class_names)}
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    elif labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        names = class_names_from_dataset(train_dir)
        mapping = {str(i): name for i, name in enumerate(names)}
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    return mapping


def _mapping_to_ordered_names(mapping: Dict[str, str]) -> List[str]:
    """Convert index-string keys to ordered class name list."""
    if not mapping:
        return []
    indices = sorted(int(k) for k in mapping.keys())
    return [mapping[str(i)] for i in indices]


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    labels_path: str = DEFAULT_LABELS_PATH,
    train_dir: str = DEFAULT_DATASET_TRAIN,
    device: torch.device | None = None,
) -> Tuple[nn.Module, List[str], torch.device]:
    """
    Load checkpoint, build model, sync class_labels.json, return model and ordered class names.

    Args:
        model_path: Path to models/best_model.pth (or checkpoint).
        labels_path: Path to inference/class_labels.json (created/updated if needed).
        train_dir: Used to generate labels if checkpoint has no class_names and JSON missing.
        device: CUDA if available unless specified.

    Returns:
        (model, class_names, device)
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp = _resolve(model_path)
    if not mp.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {mp}")

    ckpt = load_checkpoint(mp, dev)
    lp = _resolve(labels_path)
    td = _resolve(train_dir)

    if ckpt.get("class_names"):
        class_names = list(ckpt["class_names"])
        ensure_class_labels_json(lp, td, checkpoint_class_names=class_names)
    elif lp.exists():
        with open(lp, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        class_names = _mapping_to_ordered_names(mapping)
        if not class_names:
            ensure_class_labels_json(lp, td, checkpoint_class_names=None)
            with open(lp, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            class_names = _mapping_to_ordered_names(mapping)
    else:
        ensure_class_labels_json(lp, td, checkpoint_class_names=None)
        with open(lp, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        class_names = _mapping_to_ordered_names(mapping)

    model_name = str(ckpt.get("model_name", "resnet18"))
    num_classes = len(class_names)
    model = get_model(model_name, num_classes=num_classes).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, dev


def predict_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    model: nn.Module | None = None,
    class_names: List[str] | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Run inference on a single image file.

    Args:
        image_path: Path to an image file.
        model_path: Checkpoint path (used if model is None).
        model: Optional pre-loaded model (API uses this to avoid reload per request).
        class_names: Must be provided if model is provided.
        device: Used if model is provided; otherwise resolved inside load_model.

    Returns:
        {"class": class_name, "confidence": softmax probability of top class}
    """
    ip = _resolve(image_path)
    if not ip.is_file():
        raise FileNotFoundError(f"Image not found: {ip}")

    suffix = ip.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {suffix}. Use {sorted(_SUPPORTED_SUFFIXES)}")

    if model is None:
        model, class_names, device = load_model(model_path=model_path)
    else:
        if class_names is None:
            raise ValueError("class_names is required when model is provided")
        device = device or next(model.parameters()).device

    transform = get_val_test_transform()
    try:
        with Image.open(ip) as img:
            image = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid or unreadable image file: {ip}") from e

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        pred_idx = int(idx.item())
        confidence = float(conf.item())

    if pred_idx < 0 or pred_idx >= len(class_names):
        raise RuntimeError(
            f"Model output index {pred_idx} out of range for {len(class_names)} classes. "
            "Checkpoint and class_labels.json may be out of sync."
        )
    name = class_names[pred_idx]
    return {"class": name, "confidence": confidence}


def main() -> None:
    parser = argparse.ArgumentParser(description="AgroVision single-image prediction")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to checkpoint (default: models/best_model.pth)",
    )
    args = parser.parse_args()
    out = predict_image(args.image, model_path=args.model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

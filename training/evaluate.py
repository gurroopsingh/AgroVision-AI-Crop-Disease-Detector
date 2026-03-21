"""
Evaluation: accuracy, precision, recall, F1, confusion matrix on a DataLoader.
"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run model on loader (no grad), compute metrics.

    Args:
        model: Model in eval mode.
        loader: Validation or test DataLoader.
        device: Device to run on.
        class_names: Optional ordered class names (for CM labels). From loader.dataset.classes if available.

    Returns:
        Dict with: accuracy, precision, recall, f1 (weighted), confusion_matrix (ndarray),
        class_names (if provided or from dataset), and optionally per-class precision/recall/f1.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None and hasattr(loader.dataset, "classes"):
        class_names = list(loader.dataset.classes)

    return {
        "accuracy": accuracy,
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation") -> None:
    """Print metrics dict from evaluate() in a readable format."""
    print(f"\n=== {title} ===")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1 (weighted):        {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print("  Confusion matrix (sample, up to 10x10):")
    cm = metrics["confusion_matrix"]
    n = min(10, cm.shape[0])
    print(cm[:n, :n])

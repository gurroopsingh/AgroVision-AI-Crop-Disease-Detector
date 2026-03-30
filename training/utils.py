"""
Training utilities: early stopping, checkpoint save/load, metric tracking.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Stops training when the monitored metric does not improve for `patience` epochs.
    Expects higher-is-better (e.g. validation accuracy).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.epochs_without_improvement: int = 0

    def __call__(self, score: float) -> bool:
        """
        Record score and return True if training should stop.

        Args:
            score: Current epoch metric (e.g. val accuracy). Higher is better.

        Returns:
            True if early stopping triggered, else False.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def reset(self) -> None:
        """Reset state for a new training run."""
        self.best_score = None
        self.epochs_without_improvement = 0


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    best_metric: float,
    class_names: List[str],
    model_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Save training checkpoint for resuming and inference.

    Args:
        path: Output path (e.g. models/best_model.pth).
        model: Model whose state_dict to save.
        epoch: Last completed epoch.
        best_metric: Best validation metric so far.
        class_names: Ordered list of class names (for inference).
        model_name: "baseline" or "resnet18" (for loading with get_model).
        optimizer: Optional; save optimizer state for resume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "class_names": class_names,
        "model_name": model_name,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a saved checkpoint. Does not build the model; caller uses
    get_model(ckpt["model_name"], len(ckpt["class_names"])) then model.load_state_dict(ckpt["model_state_dict"]).

    Args:
        path: Path to .pth file.
        device: Device for loading (e.g. for moving state_dict later).

    Returns:
        Dict with: model_state_dict, epoch, best_metric, class_names, model_name,
        and optionally optimizer_state_dict.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    required = {"model_state_dict", "epoch", "best_metric"}
    missing = required.difference(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")
    return ckpt


class MetricTracker:
    """Stores per-epoch metric history for logging and plotting."""

    def __init__(self) -> None:
        self._history: List[Dict[str, float]] = []

    def add(self, metrics: Dict[str, float]) -> None:
        """Append one epoch's metrics (e.g. {"train_loss": 0.5, "val_acc": 0.82})."""
        self._history.append(dict(metrics))

    @property
    def history(self) -> List[Dict[str, float]]:
        """Read-only list of all epoch metrics."""
        return self._history

    def reset(self) -> None:
        """Clear history."""
        self._history.clear()

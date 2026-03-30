"""
Visualization helpers for training/evaluation artifacts.
"""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: str = "outputs/confusion_matrix.png",
) -> str:
    """
    Plot and save a confusion matrix image.

    Args:
        cm: Square confusion matrix array of shape (num_classes, num_classes).
        class_names: Ordered class labels matching matrix indices.
        output_path: File path for the output image.

    Returns:
        The saved output path as a string.
    """
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    tick_indices = np.arange(len(class_names))
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

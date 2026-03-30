"""
Training script with mixed precision, resume support, LR scheduling,
and final test-set evaluation.
"""
import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_loader import get_dataloaders
from .evaluate import evaluate, print_metrics
from .utils import EarlyStopping, MetricTracker, load_checkpoint, save_checkpoint
from .visualize import plot_confusion_matrix
from models import get_model


def get_device() -> torch.device:
    """Use CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None = None,
) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def build_limited_loader(loader: DataLoader, max_batches: int | None) -> DataLoader | Iterable:
    """Return a loader-like iterable limited to the first `max_batches` batches."""
    if max_batches is None:
        return loader
    subset_batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        subset_batches.append(batch)

    class _LimitedLoader:
        def __init__(self, batches: list[Tuple[torch.Tensor, torch.Tensor]], dataset: Any):
            self._batches = batches
            self.dataset = dataset

        def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            return iter(self._batches)

    return _LimitedLoader(subset_batches, loader.dataset)


def compute_loss_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute average loss on a loader in eval mode."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(
    model_name: str,
    base_dir: str = "data/MasterDataset",
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    num_workers: int = 4,
    best_model_path: str = "models/best_model.pth",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    resume: str | None = None,
) -> None:
    """
    Full training flow: dataloaders → model → train/val loop with early stopping and mixed precision.
    Best model (by validation accuracy) is saved to best_model_path.
    """
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=True,
    )
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    model = get_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    early_stopping = EarlyStopping(patience=patience)
    tracker = MetricTracker()

    best_val_acc = -1.0
    saved_any_checkpoint = False
    start_epoch = 0

    if resume is not None:
        ckpt = load_checkpoint(resume, device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt["best_metric"])
        saved_any_checkpoint = True
        print(
            f"Resumed from checkpoint: {resume} "
            f"(start_epoch={start_epoch + 1}, best_metric={best_val_acc:.4f})"
        )

    for epoch in range(start_epoch, epochs):
        prev_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            max_batches=max_train_batches,
        )
        val_eval_loader = build_limited_loader(val_loader, max_val_batches)
        val_metrics = evaluate(model, val_eval_loader, device, class_names=class_names)
        val_loss = compute_loss_on_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )
        val_acc = val_metrics["accuracy"]
        scheduler.step(val_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        lr_changed = new_lr != prev_lr

        tracker.add({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1_weighted": val_metrics["f1_weighted"],
            "lr": new_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=best_model_path,
                model=model,
                epoch=epoch,
                best_metric=best_val_acc,
                class_names=class_names,
                model_name=model_name,
                optimizer=optimizer,
            )
            saved_any_checkpoint = True
            print(f"  [saved best] val_acc={val_acc:.4f}")

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_metrics['f1_weighted']:.4f}  "
            f"best_val_acc={best_val_acc:.4f}  "
            f"lr={new_lr:.6f}"
        )
        if lr_changed:
            print(f"  [lr reduced] {prev_lr:.6f} -> {new_lr:.6f}")

        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")
    if saved_any_checkpoint:
        print(f"Best model saved to: {best_model_path}")
    else:
        print("No checkpoint was saved (no validation improvement).")

    if Path(best_model_path).exists():
        best_ckpt = load_checkpoint(best_model_path, device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_eval_loader = build_limited_loader(test_loader, max_test_batches)
    test_metrics = evaluate(model, test_eval_loader, device, class_names=class_names)
    print_metrics(test_metrics, title="Test Set")
    cm_path = plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["class_names"] or class_names,
    )
    print(f"Confusion matrix saved to: {cm_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgroVision training")
    parser.add_argument(
        "model_name",
        type=str,
        choices=["baseline", "resnet18"],
        help="Model to train",
    )
    parser.add_argument("--base-dir", type=str, default="data/MasterDataset", help="Dataset root")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional: limit training batches per epoch (smoke testing).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional: limit validation batches per epoch (smoke testing).",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional: limit test batches after training (smoke testing).",
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint for resuming training.",
    )
    args = parser.parse_args()

    run_training(
        model_name=args.model_name,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        best_model_path=args.best_model_path,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

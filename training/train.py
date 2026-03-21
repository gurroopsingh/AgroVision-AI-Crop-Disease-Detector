"""
Training script: full loop with validation, early stopping, mixed precision.
Saves best model to models/best_model.pth.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_loader import get_dataloaders
from .evaluate import evaluate
from .utils import EarlyStopping, save_checkpoint, MetricTracker

# Run from project root: python -m training.train
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
) -> None:
    """
    Full training flow: dataloaders → model → train/val loop with early stopping and mixed precision.
    Best model (by validation accuracy) is saved to best_model_path.
    """
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
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
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    early_stopping = EarlyStopping(patience=patience)
    tracker = MetricTracker()

    best_val_acc = -1.0
    saved_any_checkpoint = False

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            max_batches=max_train_batches,
        )
        if max_val_batches is None:
            val_metrics = evaluate(model, val_loader, device, class_names=class_names)
        else:
            subset_batches = []
            for i, batch in enumerate(val_loader):
                if i >= max_val_batches:
                    break
                subset_batches.append(batch)
            # Build a lightweight iterable with the same (images, labels) batches.
            class _TmpLoader:  # local helper, not exported
                def __init__(self, batches, dataset):
                    self._batches = batches
                    self.dataset = dataset
                def __iter__(self):
                    return iter(self._batches)
            val_metrics = evaluate(
                model,
                _TmpLoader(subset_batches, val_loader.dataset),  # type: ignore[arg-type]
                device,
                class_names=class_names,
            )
        val_acc = val_metrics["accuracy"]

        tracker.add({
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_f1_weighted": val_metrics["f1_weighted"],
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
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_metrics['f1_weighted']:.4f}  "
            f"best_val_acc={best_val_acc:.4f}"
        )

        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")
    if saved_any_checkpoint:
        print(f"Best model saved to: {best_model_path}")
    else:
        print("No checkpoint was saved (no validation improvement).")


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
        "--best-model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to save best checkpoint",
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
    )


if __name__ == "__main__":
    main()

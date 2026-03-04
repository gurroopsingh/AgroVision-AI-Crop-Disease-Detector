"""
PyTorch Dataset and DataLoader for MasterDataset (class-based image folders).
Handles corrupted images by skipping at load time; uses transforms from transforms.py.
"""
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import get_train_transform, get_val_test_transform


class CropDiseaseDataset(Dataset):
    """Dataset over class folders: root/class_name/*.jpg etc."""

    def __init__(self, root: str, transform=None, skip_corrupted: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.skip_corrupted = skip_corrupted
        self.samples = []  # list of (path, class_index)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            for path in (self.root / class_name).iterdir():
                if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    continue
                self.samples.append((str(path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            if self.skip_corrupted:
                return self.__getitem__((idx + 1) % len(self.samples))
            raise
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    base_dir: str = "data/MasterDataset",
    batch_size: int = 32,
    num_workers: int = 0,
    train_augment: bool = True,
):
    """Build train/val/test DataLoaders with 224x224 and correct transforms."""
    train_ds = CropDiseaseDataset(
        str(Path(base_dir) / "train"),
        transform=get_train_transform(augment=train_augment),
    )
    val_ds = CropDiseaseDataset(
        str(Path(base_dir) / "val"),
        transform=get_val_test_transform(),
    )
    test_ds = CropDiseaseDataset(
        str(Path(base_dir) / "test"),
        transform=get_val_test_transform(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, train_ds.classes

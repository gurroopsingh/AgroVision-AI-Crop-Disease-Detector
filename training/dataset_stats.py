"""
Dataset statistics for MasterDataset (train/val/test class folders).
Prints class counts and optionally validates images (skip corrupted).
"""   
import argparse
from pathlib import Path
from PIL import Image


def get_class_folders(root: Path):
    """Return sorted list of class folder names (direct children of root)."""
    if not root.is_dir():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def count_and_validate_images(folder: Path, validate: bool = True):
    """
    Count images in class subfolders. If validate=True, skip corrupted files.
    Returns dict: class_name -> count, and list of skipped paths.
    """
    counts = {}
    skipped = []
    for class_name in get_class_folders(folder):
        class_path = folder / class_name
        count = 0
        for path in class_path.iterdir():
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue
            if validate:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    count += 1
                except Exception:
                    skipped.append(str(path))
            else:
                count += 1
        counts[class_name] = count
    return counts, skipped


def print_dataset_stats(base_dir: str = "data/MasterDataset", validate: bool = False):
    """
    Print statistics for train/val/test under base_dir.

    Args:
        base_dir: Dataset root directory (contains train/val/test).
        validate: If True, try to open/verify each image and skip corrupted ones.
                  This is more accurate but can be slow for large datasets.
    """
    base = Path(base_dir)
    for split in ("train", "val", "test"):
        path = base / split
        if not path.exists():
            print(f"[{split}] path not found: {path}")
            continue
        counts, skipped = count_and_validate_images(path, validate=validate)
        total = sum(counts.values())
        print(f"\n--- {split} ---")
        print(f"Classes: {list(counts.keys())}")
        print(f"Per-class counts: {counts}")
        print(f"Total images: {total}")
        if validate and skipped:
            print(f"Skipped (corrupted): {len(skipped)}")
    if (base / "metadata").exists():
        print("\n--- metadata ---")
        for p in (base / "metadata").iterdir():
            print(f"  {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroVision dataset statistics")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/MasterDataset",
        help="Dataset root (contains train/val/test)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Verify images with PIL and skip corrupted files (slower).",
    )
    args = parser.parse_args()
    print_dataset_stats(base_dir=args.base_dir, validate=args.validate)

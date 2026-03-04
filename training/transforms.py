"""
Image transforms for AgroVision train/val/test pipelines.
- Train: resize, optional augmentation, then normalize.
- Val/Test: resize and normalize only.
"""
from torchvision import transforms

# ImageNet-style normalization (for transfer learning later)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transform(augment: bool = True):
    """Transform for training: resize, optional augmentation, normalize."""
    t = [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if augment:
        t.append(transforms.RandomRotation(15))
        t.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(t)


def get_val_test_transform():
    """Transform for validation/test: resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

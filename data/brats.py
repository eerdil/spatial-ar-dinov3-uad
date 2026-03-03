# datasets/brats.py

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class BraTSTrainGoodDataset(Dataset):
    """
    Train dataset: grayscale 'good' images only, converted to 3-channel for DINOv3.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.image_paths = sorted(self.img_dir.glob("*.png"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .png files found in {self.img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # img is now a 3-channel tensor [3, H, W]
        return img, str(img_path)


class BraTSAnomalyDataset(Dataset):
    """
    Validation/Test dataset for anomaly detection.

    root/
      good/
        img/*.png
        label/*.png
      Ungood/
        img/*.png
        label/*.png

    Images are grayscale but converted to 3-channel for DINOv3.
    Labels are kept as masks.
    """

    def __init__(self, root_dir, img_transform=None, label_transform=None):
        self.root_dir = Path(root_dir)
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.samples = []  # list of (img_path, label_path, target)

        for cls_name, target in [("good", 0), ("Ungood", 1)]:
            cls_root = self.root_dir / cls_name
            img_dir = cls_root / "img"
            label_dir = cls_root / "label"

            img_paths = sorted(img_dir.glob("*.png"))

            if len(img_paths) == 0:
                print(f"Warning: no images found in {img_dir}")

            for img_path in img_paths:
                label_path = label_dir / img_path.name
                if not label_path.exists():
                    raise RuntimeError(
                        f"Label not found for {img_path}: expected {label_path}"
                    )

                self.samples.append((img_path, label_path, target))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, target = self.samples[idx]

        # Open raw images (grayscale)
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)  # mask, single-channel

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)  # [3, H, W] int tensor
        else:
            # Default: convert to tensor, keep single-channel mask
            label = T.PILToTensor()(label)  # [3, H, W] int tensor

        label = (label[[0]] > 0).to(torch.uint8)

        target = torch.tensor(target, dtype=torch.long)

        meta = {
            "img_path": str(img_path),
            "label_path": str(label_path),
            "class_name": "good" if target.item() == 0 else "Ungood",
        }

        # img: [3, H, W], label: [3, H, W], target: scalar
        return img, label, target, meta


def _get_default_img_transform(img_size=224):
    """
    For grayscale medical images → 3-channel for DINOv3.
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            # T.RandomApply([
            #     T.RandomAffine(
            #         degrees=5,              # small rotations
            #         translate=(0.03, 0.03),  # small shifts
            #         scale=(0.95, 1.05),      # mild zoom
            #         shear=None,
            #     )
            # ], p=0.5),
            # T.RandomApply([
            #     T.ColorJitter(brightness=0.10, contrast=0.10)
            # ], p=0.5),
            T.ToTensor(),
            # ImageNet-style normalization (ok for DINOv3 pretrained weights)
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _get_default_label_transform(img_size=224):
    """
    For segmentation masks: resize with NEAREST to avoid mixing label values.
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),  # [1, H, W] integer mask
        ]
    )


def get_train_loader(
    dataset_root,
    train_subdir="train/good",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    img_size=224,
):
    """Return loader over the BraTS training split under ``dataset_root``."""
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / train_subdir

    img_transform = _get_default_img_transform(img_size)
    dataset = BraTSTrainGoodDataset(img_dir=img_dir, transform=img_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def get_anomaly_loader(
    dataset_root,
    split_subdir="valid",
    batch_size=32,
    shuffle=False,
    num_workers=4,
    img_size=224,
):
    """Return loader over the BraTS anomaly split under ``dataset_root``."""
    dataset_root = Path(dataset_root)
    root_dir = dataset_root / split_subdir
    img_transform = _get_default_img_transform(img_size)
    label_transform = _get_default_label_transform(img_size)
    dataset = BraTSAnomalyDataset(
        root_dir=root_dir, img_transform=img_transform, label_transform=label_transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

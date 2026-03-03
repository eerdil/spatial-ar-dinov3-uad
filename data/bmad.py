from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class BMADTrainGoodDataset(Dataset):
    """
    Train dataset: only 'good' images (PNG) from a single folder.
    Images are loaded as RGB (3-ch) for ViT/DINO-style backbones.
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

        return img, str(img_path)


class BMADAnomalyDataset(Dataset):
    """
    Validation/Test dataset:

    root/
      img/
        good/*.png
        Ungood/*.png
      label/
        (either)
          good/*.png and Ungood/*.png
        or
          *.png  (flat)

    Label is expected to have the SAME filename as the image.
    """
    def __init__(self, root_dir, img_transform=None, label_transform=None):
        self.root_dir = Path(root_dir)
        self.img_transform = img_transform
        self.label_transform = label_transform

        img_root = self.root_dir / "img"
        label_root = self.root_dir / "label"

        self.samples = []  # list of (img_path, label_path, target)

        for cls_name, target in [("good", 0), ("Ungood", 1)]:
            img_dir = img_root / cls_name
            img_paths = sorted(img_dir.glob("*.png"))

            if len(img_paths) == 0:
                print(f"Warning: no images found in {img_dir}")

            for img_path in img_paths:
                # Try label/<cls>/<name>.png first, else label/<name>.png
                label_path = (label_root / cls_name / img_path.name)
                if not label_path.exists():
                    label_path = (label_root / img_path.name)

                if not label_path.exists():
                    raise RuntimeError(
                        f"Label not found for {img_path}. Tried:\n"
                        f"  {label_root/cls_name/img_path.name}\n"
                        f"  {label_root/img_path.name}"
                    )

                self.samples.append((img_path, label_path, target))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, target = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)  # mask (could be grayscale or RGB)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)  # [1,H,W] or [C,H,W]
        else:
            label = T.PILToTensor()(label)

        # Ensure single-channel binary mask [1,H,W]
        if label.ndim == 3 and label.shape[0] > 1:
            label = label[[0]]  # take first channel
        elif label.ndim == 2:
            label = label.unsqueeze(0)

        label = (label > 0).to(torch.uint8)

        target = torch.tensor(target, dtype=torch.long)

        meta = {
            "img_path": str(img_path),
            "label_path": str(label_path),
            "class_name": "good" if target.item() == 0 else "Ungood",
        }

        return img, label, target, meta


def _get_default_img_transform(img_size=224):
    """
    For grayscale medical images → 3-channel for DINOv3.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _get_default_label_transform(img_size=224):
    """
    For segmentation masks: resize with NEAREST to avoid mixing label values.
    """
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])


def get_train_loader(dataset_root,
                     train_subdir="train/good",
                     batch_size=32,
                     shuffle=True,
                     num_workers=4,
                     img_size=224):
    """Return loader over the BraTS training split under ``dataset_root``."""
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / train_subdir

    img_transform = _get_default_img_transform(img_size)
    dataset = BMADTrainGoodDataset(img_dir=img_dir, transform=img_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader  

def get_anomaly_loader(dataset_root,
                       split_subdir="valid",
                       batch_size=32,
                       shuffle=False,
                       num_workers=4,
                       img_size=224):
    """Return loader over the BraTS anomaly split under ``dataset_root``."""
    dataset_root = Path(dataset_root)
    root_dir = dataset_root / split_subdir
    img_transform = _get_default_img_transform(img_size)
    label_transform = _get_default_label_transform(img_size)
    dataset = BMADAnomalyDataset(root_dir=root_dir,
                                  img_transform=img_transform,
                                  label_transform=label_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader

"""VisA dataset loader (all 12 categories via visa_<category> dataset name).

dataset_root points to the per-category directory (e.g. .../VisA/candle).
The category name is derived from Path(dataset_root).name.
The shared split CSV is at dataset_root/../split_csv/1cls.csv.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


def _read_split(dataset_root: Path, split: str) -> list[dict]:
    category = dataset_root.name
    split_csv = dataset_root.parent / "split_csv" / "1cls.csv"
    rows = []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["object"] == category and row["split"] == split:
                rows.append(row)
    return rows


class VisATrainGoodDataset(Dataset):
    def __init__(self, dataset_root: Path, transform=None):
        self.visa_root = dataset_root.parent
        self.rows = _read_split(dataset_root, "train")
        if not self.rows:
            raise RuntimeError(f"No train rows for '{dataset_root.name}' in split CSV")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path = self.visa_root / self.rows[idx]["image"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(img_path)


class VisAAnomalyDataset(Dataset):
    def __init__(self, dataset_root: Path, img_size: int, img_transform=None, label_transform=None):
        self.visa_root = dataset_root.parent
        self.img_size = img_size
        self.rows = _read_split(dataset_root, "test")
        if not self.rows:
            raise RuntimeError(f"No test rows for '{dataset_root.name}' in split CSV")
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.visa_root / row["image"]
        is_anomalous = row["label"] != "normal"

        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        if is_anomalous and row["mask"]:
            mask_img = Image.open(self.visa_root / row["mask"])
            if self.label_transform is not None:
                label = self.label_transform(mask_img)
            else:
                label = T.PILToTensor()(mask_img)
            label = (label[[0]] > 0).to(torch.uint8)
        else:
            label = torch.zeros(1, self.img_size, self.img_size, dtype=torch.uint8)

        target = torch.tensor(1 if is_anomalous else 0, dtype=torch.long)
        meta = {"img_path": str(img_path), "class_name": row["label"]}
        return img, label, target, meta


def _img_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _label_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])


def get_train_loader(dataset_root, img_size=448, batch_size=32, num_workers=4,
                     shuffle=True, **kwargs):
    dataset_root = Path(dataset_root)
    dataset = VisATrainGoodDataset(dataset_root=dataset_root,
                                   transform=_img_transform(img_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_anomaly_loader(dataset_root, img_size=448, batch_size=32, num_workers=4,
                       shuffle=False, **kwargs):
    dataset_root = Path(dataset_root)
    dataset = VisAAnomalyDataset(
        dataset_root=dataset_root,
        img_size=img_size,
        img_transform=_img_transform(img_size),
        label_transform=_label_transform(img_size),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

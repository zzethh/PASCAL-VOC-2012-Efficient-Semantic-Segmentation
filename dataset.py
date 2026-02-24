"""
PASCAL VOC 2012 Segmentation Dataset — HuggingFace-backed loader.
- Downloads VOC 2012 segmentation data from HuggingFace Hub
  (torchvision's Oxford mirror is frequently down)
- 80:20 train/val split from the original VOC 2012 training set
- Noise augmentations applied randomly during training for robustness
- VOC boundary class (255) mapped to ignore_index
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset

from utils import apply_random_corruption, IGNORE_INDEX


# ─────────────────── Paired Transform ───────────────────
class SegmentationTransform:
    """Joint transform for image + mask (ensures consistent spatial transforms)."""

    def __init__(self, size=300, is_train=True, noise_prob=0.5):
        self.size = size
        self.is_train = is_train
        self.noise_prob = noise_prob

        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Ensure correct modes
        image = image.convert("RGB")

        # --- Resize ---
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random colour jitter (image only) — must be before noise
            if random.random() > 0.5:
                jitter = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3
                )
                image = jitter(image)

            # Random noise corruption (image only) – builds robustness
            # Apply AFTER jitter to avoid PIL mode issues
            if random.random() < self.noise_prob:
                image = apply_random_corruption(image)
                image = image.convert("RGB")  # ensure proper mode

        # --- To tensor ---
        image = transforms.ToTensor()(image)
        image = self.img_normalize(image)  # (3, H, W)

        mask = np.array(mask, dtype=np.int64)
        # VOC uses 255 for borders / void — keep as ignore
        mask[mask == 255] = IGNORE_INDEX
        mask = torch.from_numpy(mask)  # (H, W)

        return image, mask


# ─────────────────── HuggingFace-backed Dataset ───────────────────
class VOCSegDatasetHF(Dataset):
    """
    Loads PASCAL VOC 2012 segmentation from HuggingFace Hub.
    Uses jakubkasem/pascal_voc_2012 which has proper palette (P-mode) masks.
    Columns: 'image' (PIL RGB), 'annotation' (PIL P-mode mask)
    """

    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        mask = sample["annotation"]

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


class VOCSegSubset(Dataset):
    """Subset of VocSegDatasetHF with its own transform."""

    def __init__(self, hf_dataset, indices, transform=None):
        self.data = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.data[real_idx]
        image = sample["image"]
        mask = sample["annotation"]

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


# ─────────────────── Torchvision fallback ───────────────────
class VOCSegDatasetTV(Dataset):
    """Fallback: use torchvision.datasets.VOCSegmentation if data already exists."""

    def __init__(self, root, image_set="train", download=False, transform=None):
        from torchvision.datasets import VOCSegmentation
        self.voc = VOCSegmentation(
            root=root, year="2012", image_set=image_set, download=download,
        )
        self.transform = transform

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, mask = self.voc[idx]
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


# ─────────────────── Public API ───────────────────
def get_train_val_datasets(data_root=None, size=300, noise_prob=0.5,
                           download=False, seed=42, use_hf=True):
    """
    Returns (train_dataset, val_dataset) from VOC 2012 *train* split
    with a reproducible 80:20 random split.

    Args:
        data_root: path for torchvision fallback (ignored if use_hf=True)
        use_hf: if True, download from HuggingFace Hub
    """
    train_transform = SegmentationTransform(size=size, is_train=True,
                                            noise_prob=noise_prob)
    val_transform = SegmentationTransform(size=size, is_train=False)

    if use_hf:
        print("  Loading from HuggingFace: jakubkasem/pascal_voc_2012 ...")
        hf_ds = load_dataset("jakubkasem/pascal_voc_2012", split="train")
        n = len(hf_ds)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split = int(0.8 * n)

        train_ds = VOCSegSubset(hf_ds, indices[:split], train_transform)
        val_ds = VOCSegSubset(hf_ds, indices[split:], val_transform)
    else:
        from torchvision.datasets import VOCSegmentation
        full_voc = VOCSegmentation(
            root=data_root, year="2012", image_set="train", download=download,
        )
        n = len(full_voc)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split = int(0.8 * n)

        train_ds = _TVSubset(full_voc, indices[:split], train_transform)
        val_ds = _TVSubset(full_voc, indices[split:], val_transform)

    return train_ds, val_ds


def get_test_dataset(data_root=None, size=300, use_hf=True):
    """
    Returns the VOC 2012 *validation* split as the test set.
    This is the official test set for the competition (no overlap with train).
    """
    test_transform = SegmentationTransform(size=size, is_train=False)

    if use_hf:
        # jakubkasem only has 'train'. Use Aysegul22 for val, BUT its masks are RGB.
        # Instead, use torchvision if the data exists locally, or download via alternate.
        # Best option: use the same jakubkasem dataset is only train (= VOC segmentation train).
        # For the official VOC val split, we need to try torchvision local or alternate HF.
        try:
            # Try torchvision first (if data is already downloaded)
            from torchvision.datasets import VOCSegmentation
            voc_val = VOCSegmentation(
                root=data_root, year="2012", image_set="val", download=False,
            )
            return VOCSegDatasetTV(
                root=data_root, image_set="val", download=False,
                transform=test_transform,
            )
        except Exception:
            pass

        # Fallback: tell user to download manually
        raise RuntimeError(
            "VOC 2012 val split not available. Please download manually:\n"
            "  wget https://huggingface.co/datasets/jakubkasem/pascal_voc_2012/...\n"
            "Or place VOCdevkit in the data_root directory."
        )
    else:
        return VOCSegDatasetTV(
            root=data_root, image_set="val", download=False,
            transform=test_transform,
        )


class _TVSubset(Dataset):
    """Subset of torchvision VOCSegmentation with custom transform."""

    def __init__(self, voc_dataset, indices, transform):
        self.voc = voc_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.voc[self.indices[idx]]
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

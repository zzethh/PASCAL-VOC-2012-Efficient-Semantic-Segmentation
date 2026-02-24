"""
Utility functions for PASCAL VOC semantic segmentation.
- Dice score computation (per-class and macro-averaged)
- Noise / corruption functions for robustness training & evaluation
- VOC colour palette for mask visualisation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
import random


# ─────────────────────────── VOC Constants ───────────────────────────
NUM_CLASSES = 21
IGNORE_INDEX = 255  # VOC boundary / void label

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# Standard VOC colour map (21 classes)
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


# ─────────────────────────── Dice Score ──────────────────────────────
def dice_score(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int = NUM_CLASSES,
               ignore_index: int = IGNORE_INDEX,
               smooth: float = 1e-6):
    """
    Compute per-class and macro-averaged Dice Similarity Coefficient.

    Args:
        pred:   (B, H, W) int tensor of predicted class labels
        target: (B, H, W) int tensor of ground-truth labels
    Returns:
        macro_dice: float  – macro-averaged over all 21 classes
        per_class:  list[float] – Dice for each class
    """
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    per_class = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            # Class not present in ground-truth nor prediction → perfect
            per_class.append(1.0)
        else:
            per_class.append(((2.0 * intersection + smooth) / (denom + smooth)).item())

    macro_dice = float(np.mean(per_class))
    return macro_dice, per_class


class DiceLoss(torch.nn.Module):
    """Soft Dice Loss for training (works with logits)."""

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        valid = (targets != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)

        # One-hot encode targets
        targets_clean = targets.clone()
        targets_clean[targets_clean == self.ignore_index] = 0
        one_hot = F.one_hot(targets_clean, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Mask out ignore regions
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ──────────────────── Noise / Corruption Functions ───────────────────
def add_gaussian_noise(img: Image.Image, std_range=(0.02, 0.08)):
    """Add Gaussian noise to a PIL image."""
    arr = np.array(img).astype(np.float32) / 255.0
    std = random.uniform(*std_range)
    noise = np.random.randn(*arr.shape).astype(np.float32) * std
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def add_salt_pepper_noise(img: Image.Image, amount=0.02):
    """Add salt-and-pepper noise."""
    arr = np.array(img).copy()
    h, w, c = arr.shape
    num = int(amount * h * w)
    # Salt
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    arr[ys, xs] = 255
    # Pepper
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    arr[ys, xs] = 0
    return Image.fromarray(arr)


def add_gaussian_blur(img: Image.Image, radius_range=(1, 3)):
    """Apply Gaussian blur."""
    radius = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_jpeg_compression(img: Image.Image, quality_range=(15, 40)):
    """Simulate JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_brightness_contrast(img: Image.Image, brightness_range=(0.6, 1.4),
                             contrast_range=(0.6, 1.4)):
    """Random brightness and contrast adjustment."""
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*brightness_range))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast_range))
    return img


def apply_random_corruption(img: Image.Image):
    """Apply one random corruption to an image."""
    corruption = random.choice([
        add_gaussian_noise,
        add_salt_pepper_noise,
        add_gaussian_blur,
        add_jpeg_compression,
        add_brightness_contrast,
    ])
    return corruption(img)


# ──────────────────── Mask Visualisation ────────────────────
def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Convert a class-index mask (H, W) to an RGB image using VOC palette."""
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        colour[mask == c] = VOC_COLORMAP[c]
    return Image.fromarray(colour)

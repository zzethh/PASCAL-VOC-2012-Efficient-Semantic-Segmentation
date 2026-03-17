import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
import random

NUM_CLASSES = 21
IGNORE_INDEX = 255
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def dice_score(preds, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    smooth = 1e-6
    dices, per_class = [], []
    for c in range(num_classes):
        valid = (targets != ignore_index)
        p, t = (preds == c) & valid, (targets == c) & valid
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        if union > 0:
            d = (2 * inter + smooth) / (union + smooth)
            dices.append(d.item()); per_class.append(d.item())
        else: per_class.append(0.0)
    return np.mean(dices) if dices else 0.0, per_class

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1e-6):
        super().__init__()
        self.num_classes, self.ignore_index, self.smooth = num_classes, ignore_index, smooth
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).unsqueeze(1)
        targets_clean = targets.clone()
        targets_clean[targets_clean == self.ignore_index] = 0
        one_hot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()
        probs, one_hot = probs * valid, one_hot * valid
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()

def add_gaussian_noise(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * random.uniform(0.02, 0.1), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def add_salt_pepper_noise(img):
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    num = int(random.uniform(0.01, 0.05) * h * w)
    arr[np.random.randint(0, h, num), np.random.randint(0, w, num)] = 255
    arr[np.random.randint(0, h, num), np.random.randint(0, w, num)] = 0
    return Image.fromarray(arr)

def add_gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))

def add_jpeg_compression(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(10, 40))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

def add_brightness_contrast(img):
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))).enhance(random.uniform(0.5, 1.5))

def add_grayscale(img): return img.convert("L").convert("RGB")

def add_gamma_correction(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(np.power(arr, random.uniform(0.4, 2.5)), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def add_channel_shift(img):
    arr = np.array(img).astype(np.float32)
    for c in range(3): arr[:, :, c] = np.clip(arr[:, :, c] + random.uniform(-30, 30), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def apply_random_corruption(img):
    return random.choice([add_gaussian_noise, add_salt_pepper_noise, add_gaussian_blur, add_jpeg_compression, add_brightness_contrast, add_grayscale, add_channel_shift, add_gamma_correction])(img)

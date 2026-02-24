"""
Evaluation script for PASCAL VOC 2012 semantic segmentation.
- Computes macro-averaged Dice Score over all 21 classes
- Computes FLOPs for a single forward pass
- Optionally evaluates on noisy/corrupted images
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from PIL import Image

from model import SegmentationModel, count_flops
from utils import (
    dice_score, NUM_CLASSES, IGNORE_INDEX, VOC_CLASSES,
    add_gaussian_noise, add_salt_pepper_noise,
    add_gaussian_blur, add_jpeg_compression, add_brightness_contrast,
)
from dataset import get_train_val_datasets, SegmentationTransform


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VOC segmentation model")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to best_model.pth checkpoint")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_noise", action="store_true",
                   help="Also evaluate on corrupted versions of the data")
    p.add_argument("--use_hf", action="store_true", default=True,
                   help="Use HuggingFace Hub to download dataset (default: True)")
    p.add_argument("--no_hf", dest="use_hf", action="store_false")
    return p.parse_args()


class NoisySegDataset(torch.utils.data.Dataset):
    """Wraps a Dataset to apply a specific corruption to images."""

    def __init__(self, dataset, corruption_fn, size=300):
        self.dataset = dataset
        self.corruption_fn = corruption_fn
        self.transform = SegmentationTransform(size=size, is_train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We need raw PIL to process: get_test_dataset creates subsets that apply transforms.
        # This is slightly hacky: we just fetch raw from the underlying dataset.
        if hasattr(self.dataset, "data"): # HF
            sample = self.dataset.data[idx]
            image = sample["image"]
            mask = sample["annotation"]
        else: # TV
            image, mask = self.dataset.voc[idx]
        # Apply corruption before the standard transform
        image = self.corruption_fn(image)
        image, mask = self.transform(image, mask)
        return image, mask


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    macro_d, per_class = dice_score(all_preds, all_targets)
    return macro_d, per_class


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)

    # FLOPs
    flops, params = count_flops(model, input_size=(1, 3, args.img_size, args.img_size),
                                device=device)
    flops_g = None
    print(f"\n{'='*55}")
    if flops:
        flops_g = flops / 1e9
        print(f"  Model FLOPs:  {flops_g:.4f} GFLOPs per sample")
    else:
        print("  FLOPs: N/A")
    print(f"  Model Params: {params/1e6:.4f} M" if params else "  Params: N/A")
    print(f"{'='*55}\n")

    # Dataset (use VOC val = test set)
    print("Loading VOC 2012 'val' split (20% of HF train set)...")
    try:
        _, test_ds = get_train_val_datasets(
            data_root=args.data_root, size=args.img_size, use_hf=args.use_hf
        )
        # Override the transform to test transform (no augmentations)
        test_ds.transform = SegmentationTransform(size=args.img_size, is_train=False)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
        
    print(f"  Samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Clean Evaluation ──
    print("\n─── Clean Evaluation ───")
    macro_d, per_class = evaluate(model, test_loader, device, desc="Clean Eval")

    print(f"\n{'Class':<20s} {'Dice':>8s}")
    print("-" * 30)
    for i, (name, d) in enumerate(zip(VOC_CLASSES, per_class)):
        print(f"  {name:<18s} {d:.4f}")
    print("-" * 30)
    print(f"  {'Macro Avg':<18s} {macro_d:.4f}")
    print(f"\n★ Macro-Averaged Dice Score: {macro_d:.4f}")
    if flops_g is not None:
        rank_score = macro_d / flops_g
        print(f"★ GFLOPs per sample:         {flops_g:.4f}")
        print(f"★ Ranking Metric (DICE/FLOP): {rank_score:.4f}")

    # ── Noisy Evaluation ──
    if args.test_noise:
        corruptions = {
            "Gaussian Noise": add_gaussian_noise,
            "Salt & Pepper":  add_salt_pepper_noise,
            "Gaussian Blur":  add_gaussian_blur,
            "JPEG Compress":  add_jpeg_compression,
            "Bright/Contrast": add_brightness_contrast,
        }

        print(f"\n─── Noisy / Corrupted Evaluation ───")
        for corr_name, corr_fn in corruptions.items():
            _, base_ds = get_train_val_datasets(data_root=args.data_root, size=args.img_size, use_hf=args.use_hf)
            noisy_ds = NoisySegDataset(
                base_ds,
                corruption_fn=corr_fn, size=args.img_size,
            )
            noisy_loader = DataLoader(
                noisy_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            m_d, _ = evaluate(model, noisy_loader, device, desc=corr_name)
            print(f"  {corr_name:<20s} Dice: {m_d:.4f}")


if __name__ == "__main__":
    main()

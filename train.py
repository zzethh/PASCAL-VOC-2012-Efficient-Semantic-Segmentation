"""
Training script for PASCAL VOC 2012 semantic segmentation.
- Combined CrossEntropy + Dice Loss
- AdamW optimizer with cosine annealing
- Mixed-precision training (AMP)
- Saves best model by validation Dice score
- Saves training curves
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_train_val_datasets
from model import SegmentationModel, count_flops
from utils import DiceLoss, dice_score, NUM_CLASSES, IGNORE_INDEX


def parse_args():
    p = argparse.ArgumentParser(description="Train VOC segmentation model")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Root directory for VOC dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--noise_prob", type=float, default=0.5,
                   help="Probability of applying noise augmentation per sample")
    p.add_argument("--use_hf", action="store_true", default=True,
                   help="Use HuggingFace Hub to download dataset (default: True)")
    p.add_argument("--no_hf", dest="use_hf", action="store_false",
                   help="Use torchvision local dataset instead of HuggingFace")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    p.add_argument("--save_dir", type=str, default="./checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--download", action="store_true",
                   help="Download VOC dataset if not present")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def train_one_epoch(model, loader, criterion_ce, criterion_dice, optimizer,
                    scaler, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_ce + loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Quick dice metric
        preds = logits.argmax(dim=1)
        macro_d, _ = dice_score(preds, masks)

        running_loss += loss.item()
        running_dice += macro_d
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{macro_d:.4f}")

    return running_loss / num_batches, running_dice / num_batches


@torch.no_grad()
def validate(model, loader, criterion_ce, criterion_dice, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in tqdm(loader, desc="[Val]", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_ce + loss_dice

        preds = logits.argmax(dim=1)
        macro_d, _ = dice_score(preds, masks)

        running_loss += loss.item()
        running_dice += macro_d
        num_batches += 1

    return running_loss / num_batches, running_dice / num_batches


def save_curves(history, save_dir):
    """Save training & validation loss and Dice curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o",
                 markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s",
                 markersize=3)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE + Dice)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], label="Train Dice", marker="o",
                 markersize=3)
    axes[1].plot(epochs, history["val_dice"], label="Val Dice", marker="s",
                 markersize=3)
    axes[1].set_title("Macro Dice Score per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def main():
    args = parse_args()

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    print("Loading PASCAL VOC 2012 dataset...")
    train_ds, val_ds = get_train_val_datasets(
        data_root=args.data_root,
        size=args.img_size,
        noise_prob=args.noise_prob,
        download=args.download,
        use_hf=args.use_hf,
    )
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    print("Building model: LR-ASPP MobileNetV3-Large ...")
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # FLOPs
    flops, params = count_flops(model, input_size=(1, 3, args.img_size, args.img_size),
                                device=device)
    if flops is not None:
        print(f"  FLOPs:  {flops/1e9:.3f} GFLOPs")
        print(f"  Params: {params/1e6:.2f} M")

    # Loss
    criterion_ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_dice = DiceLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}, best Dice = {best_dice:.4f}")

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    # ──────────── Training Loop ────────────
    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion_ce, criterion_dice,
            optimizer, scaler, device, epoch, args.epochs
        )

        val_loss, val_dice = validate(
            model, val_loader, criterion_ce, criterion_dice, device
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ★ New best model saved (Dice = {best_dice:.4f})")

        # Save latest every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "latest_model.pth"))

    # Save final curves and history
    save_curves(history, args.save_dir)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    if flops is not None:
        print(f"Model FLOPs: {flops/1e9:.3f} GFLOPs")
    print(f"Best model: {args.save_dir}/best_model.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

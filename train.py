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
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--noise_prob", type=float, default=0.25)
    p.add_argument("--use_hf", action="store_true", default=True)
    p.add_argument("--no_hf", dest="use_hf", action="store_false")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--download", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--model_type", type=str, default="micro_multiclass",
                   choices=["deeplabv3", "lraspp", "micro_multiclass"])
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        fg_prob = 1.0 - probs[:, 0]
        valid = (targets != IGNORE_INDEX)
        tgt_bin = (targets > 0).float() * valid.float()
        fg_prob = fg_prob * valid.float()
        intersection = (fg_prob * tgt_bin).sum()
        union = fg_prob.sum() + tgt_bin.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

class HybridCEBinaryDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = BinaryDiceLoss()

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, label_smoothing=0.1)
        bd = self.dice_loss(logits, targets)
        return ce + self.dice_weight * bd

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_dice = 0.0
        self.early_stop = False
    def __call__(self, val_dice):
        if val_dice > self.best_dice:
            self.best_dice = val_dice
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, running_dice, num_batches = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        preds_bin = (logits.argmax(dim=1) > 0).long()
        targets_bin = (masks > 0).long()
        targets_bin[masks == IGNORE_INDEX] = IGNORE_INDEX
        macro_d, _ = dice_score(preds_bin, targets_bin, num_classes=2)
        
        running_loss += loss.item()
        running_dice += macro_d
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{macro_d:.4f}")
    return running_loss / num_batches, running_dice / num_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_dice, num_batches = 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="[Val]", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        preds_bin = (logits.argmax(dim=1) > 0).long()
        targets_bin = (masks > 0).long()
        targets_bin[masks == IGNORE_INDEX] = IGNORE_INDEX
        macro_d, _ = dice_score(preds_bin, targets_bin, num_classes=2)
        running_loss += loss.item()
        running_dice += macro_d
        num_batches += 1
    return running_loss / num_batches, running_dice / num_batches

def save_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_dice"], label="Train")
    axes[1].plot(epochs, history["val_dice"], label="Val")
    axes[1].set_title("Macro Dice")
    axes[1].legend()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    train_ds, val_ds = get_train_val_datasets(data_root=args.data_root, size=args.img_size,
                                              noise_prob=args.noise_prob, download=args.download, use_hf=args.use_hf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=True, model_type=args.model_type).to(device)
    flops, params = count_flops(model, input_size=(1, 3, args.img_size, args.img_size), device=device)
    criterion = HybridCEBinaryDiceLoss(dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    start_epoch, best_dice = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch, best_dice = ckpt.get("epoch", 0) + 1, ckpt.get("best_dice", 0.0)
    
    early_stopping = EarlyStopping(patience=args.patience)
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} | Time: {time.time()-t0:.1f}s")
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice); history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(), "best_dice": best_dice},
                       os.path.join(args.save_dir, "best_model.pth"))
        if (epoch + 1) % 5 == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(), "best_dice": best_dice},
                       os.path.join(args.save_dir, "latest_model.pth"))
        if early_stopping(val_dice): break

    save_curves(history, args.save_dir)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

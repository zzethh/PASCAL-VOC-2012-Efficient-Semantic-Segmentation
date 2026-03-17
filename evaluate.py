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
from utils import (dice_score, NUM_CLASSES, IGNORE_INDEX, VOC_CLASSES, add_gaussian_noise, add_salt_pepper_noise, add_gaussian_blur, add_jpeg_compression, add_brightness_contrast, add_grayscale, add_gamma_correction, add_channel_shift)
from dataset import get_test_dataset, SegmentationTransform

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VOC segmentation model")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_noise", action="store_true")
    p.add_argument("--model_type", type=str, default="micro_multiclass", choices=["deeplabv3", "lraspp", "micro_multiclass"])
    return p.parse_args()

class NoisySegDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, corruption_fn, size=300):
        self.dataset, self.corruption_fn, self.tfm = dataset, corruption_fn, SegmentationTransform(size=size, is_train=False)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        file_name = self.dataset.file_names[idx]
        image = Image.open(os.path.join(self.dataset.images_dir, f"{file_name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.dataset.masks_dir, f"{file_name}.png"))
        return self.tfm(self.corruption_fn(image), mask)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for images, masks in tqdm(loader, desc="Eval", leave=False):
        images, masks = images.to(device), masks.to(device)
        with autocast():
            logits = model(images)
        preds_bin = (logits.argmax(dim=1) > 0).long()
        targets_bin = (masks > 0).long()
        targets_bin[masks == IGNORE_INDEX] = IGNORE_INDEX
        all_preds.append(preds_bin.cpu()); all_targets.append(targets_bin.cpu())
    return dice_score(torch.cat(all_preds), torch.cat(all_targets), num_classes=2)

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False, model_type=args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=device)["model_state"], strict=False)
    model = model.to(device)

    flops, params = count_flops(model, device=device)
    print(f"GFLOPs: {flops/1e9:.4f}" if flops else "FLOPs: N/A")
    
    test_loader = DataLoader(get_test_dataset(size=args.img_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    macro_d, per_class = evaluate(model, test_loader, device)

    print(f"\nMacro Dice: {macro_dice:.4f}")
    if flops: print(f"Ranking: {macro_d / (flops/1e9):.4f}")

    if args.test_noise:
        corruptions = {"Noise": add_gaussian_noise, "Blur": add_gaussian_blur, "JPEG": add_jpeg_compression, "Bright": add_brightness_contrast}
        for name, fn in corruptions.items():
            loader = DataLoader(NoisySegDataset(get_test_dataset(size=args.img_size), fn), batch_size=args.batch_size)
            m, _ = evaluate(model, loader, device)
            print(f"  {name}: {m:.4f}")

if __name__ == "__main__":
    main()

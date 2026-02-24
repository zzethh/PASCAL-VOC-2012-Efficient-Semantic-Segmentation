"""
Inference script for PASCAL VOC semantic segmentation.
Takes an input folder of images and produces segmentation masks.

Usage:
    python infer.py --input_folder <path> --group_number <N> --model_path best_model.pth
    
Output:
    Creates <group_number>_output/ folder containing <original_name>_mask.png files.
"""

import argparse
import os
import glob

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import SegmentationModel
from utils import NUM_CLASSES, colorize_mask
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Run segmentation inference")
    p.add_argument("--input_folder", type=str, required=True,
                   help="Folder containing test images")
    p.add_argument("--group_number", type=str, required=True,
                   help="Group number (used for output folder naming)")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to best_model.pth checkpoint")
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_colored", action="store_true",
                   help="Also save coloured visualisation masks")
    return p.parse_args()


def load_model(model_path, device):
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess(image: Image.Image, size: int):
    """Resize and normalise a PIL image → tensor (1, 3, H, W)."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@torch.no_grad()
def predict(model, image_tensor, device):
    """Run model forward pass and return predicted class mask (H, W)."""
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)  # (1, 21, H, W)
    pred = logits.argmax(dim=1).squeeze(0)  # (H, W)
    return pred.cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output folder
    output_dir = f"{args.group_number}_output"
    os.makedirs(output_dir, exist_ok=True)

    if args.save_colored:
        colored_dir = f"{args.group_number}_output_colored"
        os.makedirs(colored_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, device)

    # Collect images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_folder, ext.upper())))
    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images in {args.input_folder}")

    if len(image_paths) == 0:
        print("No images found! Check --input_folder path.")
        return

    # Run inference
    for img_path in tqdm(image_paths, desc="Inference"):
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image, args.img_size)

        mask = predict(model, tensor, device)  # (300, 300)

        # Save mask as PNG (class indices as pixel values)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = f"{basename}_mask.png"
        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(os.path.join(output_dir, mask_filename))

        # Optionally save coloured version
        if args.save_colored:
            colored = colorize_mask(mask)
            colored.save(os.path.join(colored_dir, mask_filename))

    print(f"\n✓ Segmentation masks saved to: {output_dir}/")
    print(f"  Total images processed: {len(image_paths)}")
    if args.save_colored:
        print(f"  Coloured masks saved to: {colored_dir}/")


if __name__ == "__main__":
    main()

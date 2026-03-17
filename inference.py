"""
Group 26 - PASCAL VOC 2012 Efficient Semantic Segmentation
Inference script: generates binary segmentation masks for test images.

Usage:
    python inference.py --in_dir=/path/test_images/ --out_dir=/path/26_output/

Output:
    Binary masks (foreground=255/white, background=0/black).
    Output filenames match input basenames with .png extension.
"""
import argparse
import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import SegmentationModel
from utils import NUM_CLASSES
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Group 26 - Generate binary segmentation masks")
    p.add_argument("--in_dir", type=str, required=True, help="Folder containing test images")
    p.add_argument("--out_dir", type=str, required=True, help="Folder to save output binary masks")
    p.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_model(model_path, device):
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False, model_type="micro_multiclass")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@torch.no_grad()
def predict_binary(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    pred = logits.argmax(dim=1).squeeze(0)
    binary_mask = (pred > 0).cpu().numpy().astype(np.uint8) * 255
    return binary_mask


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model_path, device)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.in_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.in_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if len(image_paths) == 0:
        print("No images found! Check --in_dir path.")
        return

    for img_path in tqdm(image_paths, desc="Inference"):
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image)
        binary_mask = predict_binary(model, tensor, device)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, f"{base_name}.png")

        mask_img = Image.fromarray(binary_mask, mode="L")
        mask_img.save(out_path)

    print(f"Done. {len(image_paths)} masks saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

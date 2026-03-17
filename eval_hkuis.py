import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as F
from model import SegmentationModel
from utils import NUM_CLASSES, dice_score, IGNORE_INDEX
from PIL import Image

def main():
    images_path = "data/HKU-IS/images.npy"
    masks_path = "data/HKU-IS/gt.npy"
    weights_path = "checkpoints/best_model.pth"

    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False, model_type="micro_multiclass").cuda()
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()

    print("Loading HKU-IS flattened arrays...")
    images_flat = np.load(images_path)
    masks_flat = np.load(masks_path)
    print(f"Loaded {len(images_flat)} samples.")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in tqdm(range(len(images_flat)), desc="Evaluating ZERO-SHOT on HKU-IS"):
            img_np = images_flat[i].reshape(256, 256, 3).astype(np.uint8)
            mask_np = masks_flat[i].reshape(256, 256).astype(np.uint8)

            img = Image.fromarray(img_np)
            img_tensor = F.to_tensor(img.resize((300, 300), Image.BILINEAR))
            img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).cuda()

            mask = Image.fromarray(mask_np)
            mask_resized = mask.resize((300, 300), Image.NEAREST)
            target_bin = torch.from_numpy((np.array(mask_resized) > 127).astype(np.int64))

            logits = model(img_tensor)
            pred_classes = logits.argmax(dim=1).squeeze().cpu()
            pred_bin = (pred_classes > 0).long()

            all_preds.append(pred_bin)
            all_targets.append(target_bin)

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    macro_dice, per_class = dice_score(all_preds, all_targets, num_classes=2)

    print("\n" + "="*50)
    print(f"ZERO-SHOT HKU-IS RESULTS")
    print("="*50)
    print(f"Global Macro Mean Dice:  {macro_dice:.4f}")
    print(f"  - Background Dice:     {per_class[0]:.4f}")
    print(f"  - Foreground Dice:     {per_class[1]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
from model import SegmentationModel
from utils import NUM_CLASSES, dice_score, IGNORE_INDEX

def main():
    voc_root = "/tmp/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    image_dir = os.path.join(voc_root, "JPEGImages")
    mask_dir = os.path.join(voc_root, "SegmentationClass")
    val_list = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    weights_path = "checkpoints/best_model.pth"

    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False, model_type="micro_multiclass").cuda()
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()

    with open(val_list, 'r') as f:
        filenames = [line.strip() for line in f]

    all_preds_bin = []
    all_targets_bin = []
    print(f"Evaluating on {len(filenames)} VOC Validation images...")

    with torch.no_grad():
        for name in tqdm(filenames, desc="Evaluating VOC-Val"):
            img_path = os.path.join(image_dir, f"{name}.jpg")
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img.resize((300, 300), Image.BILINEAR))
            img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).cuda()

            mask_path = os.path.join(mask_dir, f"{name}.png")
            mask = Image.open(mask_path)
            mask_np = np.array(mask.resize((300, 300), Image.NEAREST))

            target_bin = torch.zeros(mask_np.shape, dtype=torch.long)
            target_bin[(mask_np > 0) & (mask_np < 255)] = 1
            target_bin[mask_np == 255] = IGNORE_INDEX

            logits = model(img_tensor)
            pred_classes = logits.argmax(dim=1).squeeze().cpu()
            pred_bin = (pred_classes > 0).long()

            all_preds_bin.append(pred_bin)
            all_targets_bin.append(target_bin)

    all_preds_bin = torch.stack(all_preds_bin)
    all_targets_bin = torch.stack(all_targets_bin)
    macro_dice, per_class = dice_score(all_preds_bin, all_targets_bin, num_classes=2)

    print(f"\n{'='*40}")
    print(f"VOC-VAL OFFICIAL METRICS")
    print(f"{'='*40}")
    print(f"Global Macro Mean Dice:      {macro_dice:.4f}  (BG + FG avg)")
    print(f"  - Background Dice:         {per_class[0]:.4f}")
    print(f"  - Foreground Dice:         {per_class[1]:.4f}")
    print(f"{'='*40}")
    print(f"Ranking Metric (Dice/GFLOPS): {macro_dice / 0.0460:.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
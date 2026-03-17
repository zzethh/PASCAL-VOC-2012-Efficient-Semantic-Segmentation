import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
from model import SegmentationModel
from utils import NUM_CLASSES, dice_score, IGNORE_INDEX

def main():
    image_dir = "data/ECSSD/test/images"
    mask_dir = "data/ECSSD/test_mask/ground_truth_mask"
    weights_path = "checkpoints/best_model.pth"

    print(f"Loading model from {weights_path}...")
    model = SegmentationModel(num_classes=NUM_CLASSES, pretrained=False, model_type="micro_multiclass").cuda()
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()

    img_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Found {len(img_names)} images in ECSSD.")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for img_name in tqdm(img_names, desc="Evaluating ZERO-SHOT on ECSSD"):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((300, 300), Image.BILINEAR)
            img_tensor = F.to_tensor(img)
            img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.unsqueeze(0).cuda()

            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((300, 300), Image.NEAREST)
            mask_np = np.array(mask)
            target_bin = (mask_np > 127).astype(int)

            logits = model(img_tensor)
            pred_classes = logits.argmax(dim=1).squeeze().cpu()
            pred_bin = (pred_classes > 0).long()

            all_preds.append(pred_bin)
            all_targets.append(torch.from_numpy(target_bin))

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    macro_dice, per_class = dice_score(all_preds, all_targets, num_classes=2)

    print("\n" + "="*50)
    print(f"ZERO-SHOT ECSSD RESULTS")
    print("="*50)
    print(f"Global Macro Mean Dice:  {macro_dice:.4f}")
    print(f"  - Background Dice:     {per_class[0]:.4f}")
    print(f"  - Foreground Dice:     {per_class[1]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
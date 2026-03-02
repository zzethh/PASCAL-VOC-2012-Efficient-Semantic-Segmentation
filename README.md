# PASCAL VOC 2012 – Competition Maximised Semantic Segmentation

This repository is strictly optimised for the **PASCAL VOC 2012 Mini-Competition**. 
The official ranking metric is `DICE / FLOPS (nano)` — equivalent to **Dice / GFLOPs**.

To heavily exploit this metric, this codebase uses **LR-ASPP + MobileNetV3-Large** (~0.7 GFLOPs). Because its FLOPs are 5× lower than standard architectures like DeepLabV3, it achieves a massive multiplier on the final leaderboard. Even with slightly lower absolute accuracy, the efficiency multiplier guarantees a top-tier rank.

To compensate for the lightweight architecture on the **hidden noisy test set**, this codebase employs:
1. **Knowledge Distillation**: Transfers "dark knowledge" from a heavy DeepLabV3 teacher to the LR-ASPP student. *(Note: It is perfectly fine if the Teacher model is overfitted; it still provides robust soft class relationships and boundary smoothness for the Student to learn from).*
2. **Early Stopping & High Weight Decay**: Prevents the light model from memorising noise.
3. **Test-Time Augmentation (TTA) [OPTIONAL]**: Can be enabled (`--tta`) for maximizing pure accuracy, but is **disabled by default** to ensure the strict `DICE / FLOPs` ranking metric is not penalized by multiple forward passes.

---

## 🚀 The Winning Workflow: Exact Commands

*Please run these commands in order from the `/DATA/anikde/zenith/dl/` directory after activating your environment.*

### 0. Setup
```bash
conda activate cifar
pip install thop tqdm matplotlib
```

### 1. Optional (But Recommended): Train the Teacher Model
*Train a heavy, high-capacity model to learn the complex noise patterns. If you skip this, just run Step 2 without `--teacher_path`.*
```bash
# Trains DeepLabV3 (11M params, 3.5 GFLOPs)
python train.py --epochs 50 --batch_size 16 --lr 3e-4 --noise_prob 0.5 --gpu 0 \
    --model_type deeplabv3 --save_dir ./checkpoints_teacher
```

### 2. Required: Train the Student Model (LR-ASPP)
*This is your final submission model. It has only 0.7 GFLOPs. It will use Knowledge Distillation from the Teacher.*
```bash
# Trains LR-ASPP with distillation and early stopping
python train.py --epochs 60 --batch_size 16 --lr 3e-4 --noise_prob 0.5 --gpu 0 \
    --model_type lraspp --save_dir ./checkpoints \
    --teacher_path ./checkpoints_teacher/best_model.pth
```
*(If you skipped Step 1, simply remove the `--teacher_path ...` argument).*

### 3. Verify Your Leaderboard Score
*This reports the Macro Dice, GFLOPs per sample, and your final **Ranking Metric**.*
```bash
python evaluate.py --model_path checkpoints/best_model.pth --batch_size 16 --gpu 0 --test_noise --model_type lraspp
```

### 4. Evaluate on Instructor's 100-Sample Augmented Set
*Compare this to the previous 0.9025 baseline.*
```bash
python eval_sample.py --model_path checkpoints/best_model.pth --model_type lraspp
```

### 5. Generate Inference Demo (Submission Output)
*Generates the raw class-index PNGs required by the evaluator, recursively scanning any test folder.*
```bash
python infer.py --input_folder sample_images --group_number 26 \
    --model_path checkpoints/best_model.pth --model_type lraspp --save_colored
```

*(Note: Test-Time Augmentation (TTA) is disabled by default in `infer.py` because running multiple forward passes doubles the effective inference runtime FLOPs, which could severely penalize your final ranking score. You can enable it with `--tta` if you only care about raw Dice score).*

---

## GitHub Submission: Files to Upload
To submit this project to the evaluator, upload **only** the following files from `/DATA/anikde/zenith/dl/`:

1.  `dataset.py`
2.  `model.py`
3.  `train.py`
4.  `evaluate.py`
5.  `infer.py`
6.  `utils.py`
7.  `requirements.txt`
8.  `Dockerfile`
9.  `README.md`
10. `checkpoints/best_model.pth` *(Must be the LR-ASPP student model from Step 2!)*

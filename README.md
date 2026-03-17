# PASCAL VOC 2012 Efficient Semantic Segmentation — Group 26
**Final Submission (Phase 26)**

This repository contains our complete, submission-ready end-to-end codebase for the PASCAL VOC 2012 Efficient Semantic Segmentation mini-competition, including scripts for **training, validation, and testing**.

Our final architecture achieves a **Ranking Metric (Mean Dice / GFLOPs) of 18.57** through:
1. **Rule Compliance:** True end-to-end 21-class multi-class training. Inference output maps all 21 classes to a binary foreground/background mask without auxiliary pipelines.
2. **Nano-Skip Micro-Decoder:** Two 1×1 convolutions fuse high-level semantic features (layer 12) with low-level edge features (layer 1) from MobileNetV3-Small.
3. **Hybrid Loss:** CrossEntropy (label_smoothing=0.1) + 0.5 × Binary Dice Loss directly optimizes the evaluation metric.
4. **Resolution Scaling:** Internal 192px processing resolution mapped to the required 300px output via bilinear interpolation inside the model wrapper.

---

## 🚀 Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Inference (Generating Binary Masks)
```bash
python inference.py --in_dir=/path/to/test_images --out_dir=/path/26_output/
```
This script:
- Loads pre-trained weights from `checkpoints/best_model.pth`
- Accepts any resolution input (internally resized to 192px)
- Outputs **binary masks**: background = black (0), foreground = white (255)
- Saves output with `.png` extension for lossless quality

### 3. Training (Reproducing Results)
```bash
python train.py --epochs 150 --batch_size 16 --lr 3e-4 --noise_prob 0.25 --patience 25
```

### 4. Full Evaluation
```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

---

## 📊 Benchmarks & Metric Clarification

When evaluating binary segmentation quality, there are multiple ways to calculate the Dice score. We report **all three** for full transparency.

### Understanding the Three Dice Metrics

| Metric | Description | VOC-Val Score |
|:-------|:------------|:------------:|
| **Global Macro Mean Dice** | Mean of Background Dice and Foreground Dice across the entire dataset. This is our primary reported metric. | **0.8546** |
| **Global Foreground Dice** | Dice score computed globally over all pixels, but only for the Foreground class. | **0.7844** |
| **Image-Averaged FG Dice** | Per-image Foreground Dice averaged across all images. Highly punitive — a missed tiny object yields 0.0 for that image. | **0.6399** |

**Why is the Global Macro Mean higher?** Because background is naturally easier to predict (BG Dice = 0.9247), and the macro average of BG + FG is pulled up by the strong background performance. All three metrics are legitimate; we use the Global Macro Mean as our primary benchmark because it captures both background preservation and foreground detection.

### Final VOC-Val Results (1,449 Images)

| Metric | Score |
|:-------|:-----:|
| **Global Macro Mean Dice (BG+FG)** | **0.8546** |
| → Background Dice | 0.9247 |
| → Foreground Dice | 0.7844 |
| Image-Averaged Foreground Dice | 0.6399 |
| Average Robustness Dice (8 corruptions) | 0.8467 |
| Augmented Sample Dice | 0.8483 |
| FLOPs per sample | **0.0460 GFLOPs** |
| Parameters | 0.9395M |
| **Ranking Metric (Mean Dice / GFLOPs)** | **18.57** |

### Robustness Breakdown (8 Corruptions)

| Corruption | Dice |
|:-----------|:----:|
| Gaussian Noise | 0.8376 |
| Salt & Pepper | 0.8459 |
| Gaussian Blur | 0.8512 |
| JPEG Compression | 0.8462 |
| Brightness/Contrast | 0.8544 |
| Grayscale | 0.8519 |
| Gamma Correction | 0.8436 |
| Channel Shift | 0.8429 |

---

## 🌍 Zero-Shot Generalization (Unseen Datasets)

We evaluated the model on completely unseen saliency datasets without any fine-tuning. All metrics use the same Global Macro Mean Dice formula for consistency.

| Dataset | Images | Macro Mean Dice | Background Dice | Foreground Dice |
|:--------|:------:|:--------------:|:---------------:|:---------------:|
| **ECSSD** | 200 | **0.8478** | 0.9333 | 0.7624 |
| **HKU-IS** | 4,447 | **0.8223** | 0.9364 | 0.7082 |
| **DUTS-TE** | 5,019 | **0.8067** | 0.9419 | 0.6715 |

---

## 📈 Phase Comparison (Full Evolution)

| Phase | Student Arch | Teacher | Key Changes | FLOPs (G) | Mean Dice (BG+FG) | Aug Sample | Robustness | **Ranking** |
|:------|:-------------|:--------|:------------|:---------:|:----------:|:----------:|:----------:|:-----------:|
| Teacher | — | DeepLabV3 (MBv3-L) | Baseline | 3.506 | 0.7654 | 0.8610 | — | 0.218 |
| Teacher | — | DeepLabV3 (R101) | Baseline | 34.406 | ~0.7788 | ~0.7292 | — | 0.023 |
| Phase 1 | LR-ASPP (MBv3-L) | — | No KD | 0.736 | ~0.68 | 0.9025 | — | ~0.92 |
| Phase 3 | LR-ASPP (MBv3-L) | DeepLabV3 | KD, noise=0.5 | 0.736 | 0.7836 | 0.8075 | — | 1.065 |
| Phase 4 | LR-ASPP (MBv3-L) | DeepLabV3 | noise=0.65, α=0.3 | 0.736 | 0.7755 | 0.8031 | 0.8134 | 1.054 |
| **Phase 5** ⭐ | **LR-ASPP (MBv3-L)** | **DeepLabV3** | **Optimal noise=0.55** | **0.736** | **0.7927** | **0.8207** | **0.8284** | **1.078** |
| Phase 6 | LR-ASPP (MBv3-L) | DeepLabV3 | Scale+Crop, weights | 0.736 | 0.7762 | — | 0.7787 | 1.055 |
| Phase 7 | LR-ASPP (MBv3-L) | ResNet101 | Kaggle Leak Fix | 0.402 | 0.7365 | 0.7292 | 0.7137 | 1.834 |
| Phase 8 | LR-ASPP (MBv3-L) | ResNet101 | OHEM, PolyLR | 0.402 | 0.6722 | 0.6159 | 0.6452 | 1.673 |
| Phase 9 | Fused LR-ASPP | ResNet101 | Conv-BN Fusion | 0.402 | 0.5805 | 0.4861 | — | 1.445 |
| Phase 11 | LR-ASPP (MBv3-S) | ResNet101 | Width Scaling | 0.121 | 0.5889 | 0.5815 | 0.5696 | 4.869 |
| Phase 12 | LR-ASPP (MBv3-S) | ResNet101 | Long Bake 150ep | 0.121 | 0.5841 | 0.5537 | 0.5651 | 4.830 |
| Phase 13 | LR-ASPP (MBv3-L) | ResNet101 | KD Clamp + Compound | 0.402 | 0.7265 | 0.6999 | 0.7005 | 1.809 |
| Phase 14 | LR-ASPP (MBv3-L) | ResNet101 | KD Clamp Only | 0.402 | 0.7328 | 0.6666 | 0.7041 | 1.825 |
| **Phase 15** | **LR-ASPP (MBv3-L)** | **DeepLabV3** | **DeepLabV3 teacher** | **0.402** | **0.7874** | **0.7775** | **0.7631** | **1.960** |
| Phase 16 | LR-ASPP (MBv3-S) | DeepLabV3 | MBv3-Small + DLv3 | 0.121 | 0.6055 | 0.5739 | 0.5801 | 5.007 |
| Phase 17 | LR-ASPP (MBv3-S) | DeepLabV3 | 192px internal | 0.089 | 0.5616 | 0.5571 | 0.5414 | 6.312 |
| Phase 18 | LR-ASPP (MBv3-S) | DeepLabV3 | Retrain of Ph17 | 0.089 | 0.5571 | 0.5411 | 0.5394 | 6.259 |
| Phase 19 🚀 | LR-ASPP (MBv3-S) | None | Binary Target Simplify | 0.089 | 0.8469 | 0.8319 | 0.8359 | 9.515 |
| Phase 20 | Nano-Skip (MBv3-S) | None | 1×1 Conv Fusion (2ch) | 0.045 | 0.8421 | 0.8304 | 0.8334 | 18.742 |
| Phase 23 | Nano-Skip (MBv3-S) | None | 21-Class Compliant | 0.046 | 0.7987 | 0.7795 | 0.7841 | 17.35 |
| *Phase 24* ❌ | *Nano-Skip (MBv3-S)* | *DeepLabV3* | *KD (aborted)* | *0.046* | *~0.77* | *—* | *—* | *~16.7* |
| Phase 25 ✅ | Nano-Skip (MBv3-S) | None | Verification of Ph23 | 0.046 | 0.7987 | 0.7795 | 0.7839 | 17.35 |
| **Phase 26** 🏆 | **Nano-Skip (MBv3-S)** | **None** | **CE+BinaryDice+LS, 150ep** | **0.046** | **0.8546** | **0.8483** | **0.8467** | **18.57** |

> **Ranking Metric** = Mean Dice / GFLOPs. Higher is better.
> **Current Best Legal Submission: Phase 26** (18.57 ranking, 21-class compliant).
> Phases 19–20 had higher rankings but used illegal binary-only training (disqualified under new rules).

---

## 📂 Codebase Structure

| File | Purpose |
|:-----|:--------|
| `inference.py` | Generates binary masks for test images (accepts `--in_dir` and `--out_dir`) |
| `train.py` | Training script with HybridCEBinaryDiceLoss, CosineAnnealingLR, AMP |
| `dataset.py` | PASCAL VOC 2012 loader via Kaggle with strict train/val separation |
| `model.py` | Nano-Skip Micro-Decoder architecture (MobileNetV3-Small backbone) |
| `utils.py` | Dice score, augmentations, loss definitions |
| `evaluate.py` | Full validation with robustness and augmented sample testing |
| `eval_voc_val.py` | VOC-Val evaluation reporting Global Mean, Global FG, and Image-Averaged Dice |
| `eval_ecssd.py` | Zero-shot ECSSD evaluation (200 images) |
| `eval_duts.py` | Zero-shot DUTS-TE evaluation (5,019 images) |
| `eval_hkuis.py` | Zero-shot HKU-IS evaluation (4,447 images) |
| `checkpoints/best_model.pth` | Final trained weights |

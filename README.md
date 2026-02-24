# PASCAL VOC 2012 – Efficient Semantic Segmentation

Lightweight semantic segmentation on PASCAL VOC 2012 using **LR-ASPP with MobileNetV3-Large backbone**, optimised for the best **Dice Score ↔ FLOPs** trade-off.

## Model Architecture

| Property | Value |
|---|---|
| Backbone | MobileNetV3-Large (ImageNet pretrained) |
| Head | LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling) |
| Input | `(3, 300, 300)` RGB |
| Output | `(300, 300)` mask, 21 classes `[0–20]` |
| Params | ~3.2 M |
| FLOPs | ~0.7 GFLOPs per image |

## GitHub Submission: Files to Upload

To submit this project, upload the following files from `/DATA/anikde/zenith/dl/`:

1.  **`dataset.py`**: Custom data loader using HuggingFace Hub (robust to original server outages).
2.  **`model.py`**: LR-ASPP MobileNetV3 architecture definition.
3.  **`train.py`**: Training engine (AMP, Cosine Annealing, combined loss).
4.  **`evaluate.py`**: Standardized evaluation (Cleaning & Noisy Dice, FLOPs calculation).
5.  **`infer.py`**: Prediction script for competition masks.
6.  **`utils.py`**: Core metrics (Dice Score), Dice Loss, and Noise artifacts.
7.  **`requirements.txt`**: Python dependencies.
8.  **`Dockerfile`**: Containerized environment definition.
9.  **`README.md`**: Project documentation (this file).
10. **`checkpoints/best_model.pth`**: Trained model weights (required for the demo/evaluation).

## Workflow: How the Assignment Requirements were Achieved

1.  **Model Selection**: Chose **LR-ASPP with MobileNetV3-Large** because it is specifically designed for high-efficiency segmentation. This ensures a high **DICE/FLOPs** ratio.
2.  **Robust Data Loading**: Replaced the frequently-down official VOC mirrors with a reliable HuggingFace dataset loader. This ensures the project is reproducible anywhere.
3.  **Data Split**: Implemented a deterministic 80/20 train/validation split of the VOC training data to ensure no overlap during training.
4.  **Training for Robustness**: integrated random **Gaussian Noise, Salt-and-Pepper, Blur, and Compression** directly into the training loop. This forces the model to learn features invariant to common image corruptions.
5.  **Combined Loss**: Used **CrossEntropy + Dice Loss**. CrossEntropy provides stable gradients, while Dice Loss directly optimizes the competition metric and handles class imbalance.
6.  **Efficiency Optimization**: Enabled **AMP (Automatic Mixed Precision)** and high-speed data loading to minimize training time on the available GTX 1080 Ti GPUs.
7.  **Standardized Inference**: Created a strict inference script that handles arbitrary resolution inputs and outputs naming-compliant PNG masks.

## Quick Start: Exact Commands

### 1. Setup & Environment
```bash
conda activate cifar
pip install thop tqdm matplotlib
```

### 2. Training (Standard Run)
This command will train for 50 epochs and save the best weights to `checkpoints/best_model.pth`.
```bash
python train.py --epochs 50 --batch_size 16 --lr 3e-4 --noise_prob 0.5 --gpu 0 --use_hf
```

### 3. Evaluation (Metrics & FLOPs)
Reports Macro Dice, GFLOPs per sample, and the DICE/FLOPs ranking metric to 4 decimal places.
```bash
python evaluate.py --model_path checkpoints/best_model.pth --batch_size 16 --gpu 0 --use_hf --test_noise
```

### 4. Inference Demo (Group 26)
Generates submission-ready masks for images in a folder.
```bash
python infer.py --input_folder sample_images --group_number 26 --model_path checkpoints/best_model.pth --save_colored
```

---

## How to Perform a Demo

To quickly demonstrate the model's performance on the provided sample images:

1.  **Activate Env**: `conda activate cifar`
2.  **Run Inference**:
    ```bash
    python infer.py --input_folder sample_images --group_number 26 --model_path checkpoints/best_model.pth --save_colored
    ```
3.  **Check Results**:
    - `26_output/`: Contains the raw class-index masks (pixel values 0–20).
    - `26_output_colored/`: Contains the visual palette masks.

---

## Project Structure

```
zenith/dl/
├── train.py          # Training loop (AMP, cosine LR, checkpointing)
├── evaluate.py       # Dice score + FLOPs evaluation + robustness testing
├── infer.py          # Folder-based inference → mask output
├── model.py          # LR-ASPP MobileNetV3-Large wrapper
├── dataset.py        # VOC 2012 dataset + 80:20 split + augmentations
├── utils.py          # Dice metric, Dice loss, noise functions, VOC palette
├── requirements.txt  # Dependencies
├── Dockerfile        # Container build
└── README.md         # This file
```

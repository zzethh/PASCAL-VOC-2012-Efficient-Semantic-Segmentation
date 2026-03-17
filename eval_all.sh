#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv

MODEL_PATH="./checkpoints/best_model.pth"

echo "=== FINAL EVALUATION ==="

echo "[1/2] Running Clean Evaluation & Robustness..."
python evaluate.py --model_path $MODEL_PATH --test_noise > eval_res.txt 2>&1

echo "[2/2] Running Augmented Sample Evaluation..."
python eval_sample.py --model_path $MODEL_PATH > eval_sample_res.txt 2>&1

echo "Evaluation complete!"
echo "Results:"
grep "Macro-Averaged Dice Score" eval_res.txt
grep "Avg Robustness" eval_res.txt
grep "Macro-Averaged Dice Score on Sample Set" eval_sample_res.txt

#!/bin/bash

#SBATCH --job-name=eval-lora-ckpt
#SBATCH -D .
#SBATCH -A large-sc-2
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --environment=lsaie3_vllm

# ----------------------------
# Project paths
# ----------------------------
export PROJECT_FOLDER=$SCRATCH/LSAIE/project/peft_apertus/lexam
export PYTHON_FILE=$PROJECT_FOLDER/eval_lora_checkpoint.py
export CONFIG_FILE=$PROJECT_FOLDER/config_lora.yaml

# ----------------------------
# LoRA checkpoint to evaluate (SET THIS)
# ----------------------------
export LORA_PATH=$SCRATCH/LSAIE/project/results/grid_8B_lora/lr1e-3_r128_lora_8B/checkpoint-100

# ----------------------------
# Run evaluation
# ----------------------------
srun bash -c "
python3 $PYTHON_FILE \
    --config \"$CONFIG_FILE\" \
    --lora-path \"$LORA_PATH\" \
    --max-lora-rank 512
"

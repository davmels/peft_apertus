#!/bin/bash

#SBATCH --job-name=eval-grid-70B-lora_lr1e-3
#SBATCH -D .
#SBATCH -A large-sc-2
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --environment=lsaie3_vllm

# ----------------------------
# Project paths
# ----------------------------
export PROJECT_FOLDER=$SCRATCH/LSAIE/project/peft_apertus/lexam
export PYTHON_FILE=$PROJECT_FOLDER/eval_all_checkpoints.py
export CONFIG_FILE=$PROJECT_FOLDER/config_70b.yaml

# Grid to evaluate
export GRID_DIR=$SCRATCH/LSAIE/project/results/grid_70B_lora_lr1e-3

# ----------------------------
# Run evaluation on all checkpoints
# ----------------------------
srun bash -c "
python3 $PYTHON_FILE \
    --config \"$CONFIG_FILE\" \
    --grid-dir \"$GRID_DIR\" \
    --checkpoint-name checkpoint-467 \
    --max-lora-rank 512 \
    --tensor-parallel-size 4
"

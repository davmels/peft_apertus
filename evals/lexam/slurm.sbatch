#!/bin/bash

#SBATCH --job-name=eval-lexam-vllm
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

# # ----------------------------
# # Project paths
# # ----------------------------
export PROJECT_FOLDER=$SCRATCH/LSAIE/project/peft_apertus/lexam
export PYTHON_FILE=$PROJECT_FOLDER/eval_lexam_vllm.py

# # Choose config here:
export CONFIG_FILE=$PROJECT_FOLDER/config.yaml
# export CONFIG_FILE=$PROJECT_FOLDER/config_70b.yaml

# ----------------------------
# Run evaluation
# ----------------------------
srun bash -c "
python3 $PYTHON_FILE \
    --config \"$CONFIG_FILE\"
"
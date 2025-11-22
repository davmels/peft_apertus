#!/bin/bash

#SBATCH --job-name=sft-apertus
#SBATCH -D .
#SBATCH -A large-sc-2
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=00:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=lsaie3      # using compressed docker image as an environment

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$HF_TOKEN
######################

pip install --user --upgrade pip
pip install --user git+https://github.com/huggingface/transformers.git 
unset SSL_CERT_FILE
pip install --user --upgrade --no-deps datasets

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_MACHINES=$SLURM_NNODES
export NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

export CONFIG_FILE="$SCRATCH/LSAIE/project/peft_apertus/configs/ddp.yaml"
export PYTHON_FILE="$SCRATCH/LSAIE/project/peft_apertus/sft_train.py"
export SFT_CONFIG="$SCRATCH/LSAIE/project/peft_apertus/configs/sft_lora.yaml"

echo "Master Node: $MASTER_ADDR"
echo "Num Machines: $NUM_MACHINES"
echo "Total Processes: $NUM_PROCESSES"

cd $SCRATCH/LSAIE/project/peft_apertus/

export LAUNCH_CMD="accelerate launch \
    --config_file $CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $NUM_PROCESSES \
    --num_machines $NUM_MACHINES \
    --machine_rank \$SLURM_PROCID \
    $PYTHON_FILE \
    --config $SFT_CONFIG"

echo "Running command template: $LAUNCH_CMD"

echo "Running command: $LAUNCH_CMD"

srun bash -c "$LAUNCH_CMD"

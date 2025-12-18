#!/bin/bash

#SBATCH --job-name=sft-apertus
#SBATCH -D .
#SBATCH -A large-sc-2
#SBATCH --output=./logs_new/O-%x.%j
#SBATCH --error=./logs_new/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --environment=lsaie3


export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600


export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$HF_TOKEN


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_MACHINES=$SLURM_NNODES
export NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))


export CONFIG_FILE="$SCRATCH/LSAIE/project/peft_apertus/configs/zero3_multinode.yaml"
export PYTHON_FILE="$SCRATCH/LSAIE/project/peft_apertus/sft_train.py"
export SFT_CONFIG="$SCRATCH/LSAIE/project/peft_apertus/configs/sft_lora.yaml"

echo "Master Node: $MASTER_ADDR"
echo "Num Machines: $NUM_MACHINES"
echo "Total Processes: $NUM_PROCESSES"

cd $SCRATCH/LSAIE/project/peft_apertus/
echo "Current Directory: $(pwd)"

srun bash -c '
echo "Node $(hostname) - Rank $SLURM_PROCID - Connecting to $MASTER_ADDR"
pip install --upgrade git+https://github.com/huggingface/transformers.git 
unset SSL_CERT_FILE
pip install --user --upgrade --no-deps datasets

accelerate launch \
    --config_file "$CONFIG_FILE" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$NUM_MACHINES" \
    --machine_rank "$SLURM_PROCID" \
    "$PYTHON_FILE" \
    --config "$SFT_CONFIG"
'
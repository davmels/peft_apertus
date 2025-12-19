#!/bin/bash

#SBATCH --job-name=inf_grid
#SBATCH -D .
#SBATCH -A large-sc-2
#SBATCH --output=logs/full/inf_grid_%x_%j.out
#SBATCH --error=logs/full/inf_grid_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=288    
#SBATCH --time=03:45:00
#SBATCH --environment=lsaie_vllm

# --- 1. Environment & Network Setup ---
# export NCCL_SOCKET_IFNAME=^lo,docker0
# export NCCL_DEBUG=INFO 
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$HF_TOKEN

export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"


# --- 2. Argument Parsing ---
# We use the arguments passed from the launcher script
LORA_PATH=$1
OUTPUT_DIR=$2
BASE_MODEL=$3

if [ -z "$LORA_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Arguments missing. Usage: sbatch inference_grid.sbatch <LORA_PATH> <OUTPUT_DIR> <BASE_MODEL>"
  exit 1
fi

# Exporting these so they are available inside 'srun'
export LORA_PATH
export OUTPUT_DIR
export BASE_MODEL
export PYTHON_SCRIPT="$SCRATCH/LSAIE/project/peft_apertus/evals/inference_lora.py"
export DATASET_PATH="rcds/swiss_judgment_prediction" #"$SCRATCH/LSAIE/project/peft_apertus/datasets/swiss_judgment_prediction_dataset"

echo "========================================"
echo "Job: Inference on 8B Grid"
echo "Base Model: $BASE_MODEL"
echo "Adapter:    $LORA_PATH"
echo "Output:     $OUTPUT_DIR"
echo "========================================"

# --- 3. Execution ---
# We cd to project root to ensure relative imports work if needed
cd $SCRATCH/LSAIE/project/peft_apertus/

srun bash -c '
python "$PYTHON_SCRIPT" \
    --base_model_name_or_path="$BASE_MODEL" \
    --dataset_name="$DATASET_PATH" \
    --dataset_config="all" \
    --lora_adapter_path="$LORA_PATH" \
    --split=test \
    --input_col=text \
    --output_col=generated_answer \
    --processing_function=swiss_judgement_prediction \
    --max_new_tokens=200 \
    --temperature=0.0 \
    --output_dir="$OUTPUT_DIR" \
    --start_index=0 \
    --load_adapter_as_model
'
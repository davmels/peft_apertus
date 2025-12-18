#!/bin/bash

# Define directories
GRID_ROOT="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_8B_lora"
OUTPUT_ROOT="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/evals/inference_results/grid_8B_lora"
BASE_MODEL="swiss-ai/Apertus-8B-Instruct-2509"  # Ensure this is the correct 8B base model

# Check if grid root exists
if [ ! -d "$GRID_ROOT" ]; then
    echo "Error: Directory $GRID_ROOT does not exist."
    exit 1
fi

# Iterate over each model folder in the grid
for model_path in "$GRID_ROOT"/*; do
    if [ -d "$model_path" ]; then
        model_name=$(basename "$model_path")
        
        # Define where to save the inference results for this specific model
        save_path="$OUTPUT_ROOT/$model_name"
        
        echo "Submitting inference for: $model_name"
        echo "  - LoRA Path: $model_path"
        echo "  - Output Dir: $save_path"

        # Submit the SLURM job, passing arguments (LORA_PATH, OUTPUT_DIR, BASE_MODEL)
        # We use --job-name to make it easy to track specific runs in squeue
        sbatch --job-name="inf_${model_name}" \
               inference_grid.sbatch \
               "$model_path" \
               "$save_path" \
               "$BASE_MODEL"
               
        sleep 1 # Brief pause to prevent overloading the scheduler
    fi
done
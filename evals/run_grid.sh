#!/bin/bash

# ================= CONFIGURATION =================
# 1. The Grid (Space separated)
LEARNING_RATES=("1e-5")
# LORA_RS=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512")
LORA_RS=("1")
USE_PEFT=false

# 2. Fixed Parameters
MODEL_NAME="swiss-ai/Apertus-70B-Instruct-2509"
BATCH_SIZE="4"
GRAD_ACCUM="2"
NNODES="4"

# 3. Paths
TEMPLATE_FILE="$SCRATCH/LSAIE/project/peft_apertus/slurm.sbatch"
BASE_OUTPUT_DIR="$SCRATCH/LSAIE/project/peft_apertus/results/grid_70B_full"
DATASET_PATH="rcds/swiss_judgment_prediction"
PROCESSING_STRATEGY="swiss_judgment_prediction"
DATASET_CONFIG="all"
SFT_CONFIG="$SCRATCH/LSAIE/project/peft_apertus/configs/sft_full70.yaml"
ACCELERATOR_CONFIG="$SCRATCH/LSAIE/project/peft_apertus/configs/zero3_multinode.yaml"
# =================================================

mkdir -p "$BASE_OUTPUT_DIR"

echo "Starting Grid Search..."

RUNNING_JOBS=$(squeue -u $USER -h -o "%j")

for LR in "${LEARNING_RATES[@]}"; do
    for R in "${LORA_RS[@]}"; do
        
        RUN_ID="lr${LR}_r${R}"

        if [ "$USE_PEFT" = true ]; then
            RUN_ID="${RUN_ID}_lora"
        else 
            RUN_ID="${RUN_ID}_full"
        fi

        if [ $MODEL_NAME == "swiss-ai/Apertus-8B-Instruct-2509" ]; then
            RUN_ID="${RUN_ID}_8B"
        elif [ $MODEL_NAME == "swiss-ai/Apertus-70B-Instruct-2509" ]; then
            RUN_ID="${RUN_ID}_70B"
        fi

        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"
        JOB_NAME="sft-${RUN_ID}"

        # --- 1. CHECK IF JOB IS RUNNING ---
        # We verify if our constructed JOB_NAME is in the list of running jobs
        if echo "$RUNNING_JOBS" | grep -q "^${JOB_NAME}$"; then
            echo "[SKIP] Job is currently running: ${RUN_ID}"
            continue
        fi

        # --- 2. CHECK IF COMPLETED OUTPUT EXISTS ---
        # If the directory exists, check if it contains valid model files (.json or .safetensors)
        if [ -d "$OUTPUT_DIR" ]; then
            # Look for at least one relevant file type
            if ls "$OUTPUT_DIR"/*.json 1> /dev/null 2>&1 || ls "$OUTPUT_DIR"/*.safetensors 1> /dev/null 2>&1; then
                echo "[SKIP] Found valid completed output: ${RUN_ID}"
                continue
            else
                echo "[OVERWRITE] Output dir exists but seems empty/incomplete. Resubmitting: ${RUN_ID}"
            fi
        fi
        
        # Flattened extra args (single line is safer for sed injection)
        EXTRA_ARGS="--learning_rate ${LR} --lora_r ${R} --use_peft ${USE_PEFT} --output_dir ${OUTPUT_DIR} --model_name_or_path ${MODEL_NAME} --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRAD_ACCUM} --processing_strategy ${PROCESSING_STRATEGY} --dataset_name ${DATASET_PATH} --dataset_config ${DATASET_CONFIG} --run_name ${RUN_ID}"

        echo "Submitting run: ${RUN_ID}"

        # 1. Capture the modified script into a variable
        FINAL_SCRIPT=$(sed -e "s/#SBATCH --job-name=.*/#SBATCH --job-name=${JOB_NAME}/" \
            -e "s|#SBATCH --output=.*|#SBATCH --output=./logs_grid/O-${RUN_ID}.%j|" \
            -e "s|#SBATCH --error=.*|#SBATCH --error=./logs_grid/E-${RUN_ID}.%j|" \
            -e "s|#SBATCH --nodes=.*|#SBATCH --nodes=${NNODES}|" \
            -e "s|--config_file \"\$CONFIG_FILE\"|--config_file \"${ACCELERATOR_CONFIG}\"|" \
            -e "s|--config \"\$SFT_CONFIG\"|--config \"${SFT_CONFIG}\" ${EXTRA_ARGS}|" \
            "$TEMPLATE_FILE")

        # 2. Print it to the console (DEBUGGING)
        echo "================= PREVIEW: ${RUN_ID} ================="
        echo "$FINAL_SCRIPT"
        echo "======================================================"

        # 3. Submit the variable content to sbatch
        echo "$FINAL_SCRIPT" | sbatch

        sleep 0.2
    done
done
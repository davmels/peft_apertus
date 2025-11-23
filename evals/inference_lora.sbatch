#!/bin/bash

#SBATCH --job-name=inference_70b
#SBATCH -A large-sc-2
#SBATCH --output=logs/inf_%j.out
#SBATCH --error=logs/inf_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --environment=lsaie3

export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$HF_TOKEN

pip install --user --upgrade pip
pip install --user git+https://github.com/huggingface/transformers.git 
unset SSL_CERT_FILE
pip install --user --upgrade --no-deps datasets

# --lora_adapter_path=LSAIE/project/peft_apertus/models/lora_apertus_70b \
python evals/inference_lora.py --base_model_name_or_path=swiss-ai/Apertus-70B-Instruct-2509 \
    --dataset_name=$SCRATCH/LSAIE/project/peft_apertus/datasets/swiss_judgment_prediction_dataset \
    --lora_adapter_path=$SCRATCH/LSAIE/project/peft_apertus/models/apertus70B_sft_lora \
    --split=test \
    --input_col=text \
    --output_col=generated_answer \
    --processing_function=swiss_judgement_prediction \
    --max_new_tokens=10 \
    --temperature=0.0 \
    --output_dir=$SCRATCH/LSAIE/project/peft_apertus/evals/inference_results/70b_base_instruct_sft_test_6 \
    --start_index=7500
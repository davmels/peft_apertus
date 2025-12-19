#!/bin/bash
# Script to convert DeepSpeed ZeRO sharded checkpoint to HuggingFace format

# === CONFIGURATION ===
# Set these paths according to your setup
CHECKPOINT_DIR="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_70B_full/lr1e-4_r1_full_70B/checkpoint-467"  # The checkpoint folder with global_step466
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_70B_full_hf_models"          # Where to save the converted model

# === CONVERSION ===
echo "Converting DeepSpeed ZeRO checkpoint to FP32..."
echo "Input: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the conversion script
# This consolidates all the sharded weights into a single pytorch_model.bin
python "$CHECKPOINT_DIR/zero_to_fp32.py" "$CHECKPOINT_DIR" "$OUTPUT_DIR/pytorch_model.bin"

# Copy config files needed for HuggingFace
echo "Copying config files..."
cp "$CHECKPOINT_DIR/config.json" "$OUTPUT_DIR/"
cp "$CHECKPOINT_DIR/tokenizer.json" "$OUTPUT_DIR/"
cp "$CHECKPOINT_DIR/tokenizer_config.json" "$OUTPUT_DIR/"
cp "$CHECKPOINT_DIR/generation_config.json" "$OUTPUT_DIR/"
cp "$CHECKPOINT_DIR/chat_template.jinja" "$OUTPUT_DIR/" 2>/dev/null || true

echo "Done! Model saved to: $OUTPUT_DIR"
echo ""
echo "You can now load it with:"
echo "  from transformers import AutoModelForCausalLM"
echo "  model = AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR')"
echo ""
echo "Or with vLLM:"
echo "  from vllm import LLM"
echo "  llm = LLM(model='$OUTPUT_DIR')"

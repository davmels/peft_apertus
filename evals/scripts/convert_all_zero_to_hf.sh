#!/bin/bash
# Script to batch convert all DeepSpeed ZeRO checkpoints to HuggingFace format

# === CONFIGURATION ===
INPUT_BASE_DIR="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_70B_full"
OUTPUT_BASE_DIR="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_70B_full_hf_models"

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "========================================"
echo "Batch DeepSpeed ZeRO to HuggingFace Converter"
echo "========================================"
echo "Input directory: $INPUT_BASE_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo ""

# Find all model directories (those containing zero_to_fp32.py)
for model_dir in "$INPUT_BASE_DIR"/*/; do
    model_name=$(basename "$model_dir")
    
    # Check if this is a valid model directory
    if [ ! -f "$model_dir/zero_to_fp32.py" ]; then
        echo "Skipping $model_name - not a DeepSpeed checkpoint"
        continue
    fi
    
    echo "========================================"
    echo "Processing: $model_name"
    echo "========================================"
    
    # Find the latest checkpoint folder (highest number)
    latest_checkpoint=$(ls -d "$model_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    
    if [ -z "$latest_checkpoint" ]; then
        echo "  ERROR: No checkpoint-* folder found in $model_dir"
        continue
    fi
    
    checkpoint_name=$(basename "$latest_checkpoint")
    echo "  Using checkpoint: $checkpoint_name"
    
    # Create output directory for this model
    output_dir="$OUTPUT_BASE_DIR/$model_name"
    mkdir -p "$output_dir"
    
    # Check if already converted
    if [ -f "$output_dir/pytorch_model.bin" ] || [ -f "$output_dir/model.safetensors" ]; then
        echo "  SKIP: Already converted (output exists)"
        continue
    fi
    
    echo "  Converting to: $output_dir"
    
    # Run the conversion - use the zero_to_fp32.py from the checkpoint folder
    # The script expects: python zero_to_fp32.py <checkpoint_dir> <output_file>
    python "$latest_checkpoint/zero_to_fp32.py" "$latest_checkpoint" "$output_dir/pytorch_model.bin"
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Conversion failed for $model_name"
        continue
    fi
    
    # Copy config files
    echo "  Copying config files..."
    cp "$latest_checkpoint/config.json" "$output_dir/" 2>/dev/null || cp "$model_dir/config.json" "$output_dir/"
    cp "$latest_checkpoint/tokenizer.json" "$output_dir/" 2>/dev/null || cp "$model_dir/tokenizer.json" "$output_dir/"
    cp "$latest_checkpoint/tokenizer_config.json" "$output_dir/" 2>/dev/null || cp "$model_dir/tokenizer_config.json" "$output_dir/"
    cp "$latest_checkpoint/generation_config.json" "$output_dir/" 2>/dev/null || cp "$model_dir/generation_config.json" "$output_dir/"
    cp "$latest_checkpoint/chat_template.jinja" "$output_dir/" 2>/dev/null || cp "$model_dir/chat_template.jinja" "$output_dir/" 2>/dev/null || true
    
    echo "  ✓ Done: $model_name"
    echo ""
done

echo "========================================"
echo "All conversions complete!"
echo "Models saved to: $OUTPUT_BASE_DIR"
echo "========================================"
ls -la "$OUTPUT_BASE_DIR"

#!/bin/bash
# Script to fix the directory structure of converted models
# The sharded files are inside pytorch_model.bin/ folder, but HF expects them in the root

OUTPUT_BASE_DIR="/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/results/grid_70B_full_hf_models"

echo "Fixing model directory structure..."
echo ""

for model_dir in "$OUTPUT_BASE_DIR"/*/; do
    model_name=$(basename "$model_dir")
    echo "Processing: $model_name"
    
    # Check if pytorch_model.bin is a directory (wrong structure)
    if [ -d "$model_dir/pytorch_model.bin" ]; then
        echo "  Found incorrect structure: pytorch_model.bin is a directory"
        
        # Check if there's an index file or sharded files inside
        if ls "$model_dir/pytorch_model.bin/"*.bin 1> /dev/null 2>&1; then
            echo "  Moving sharded files to root..."
            mv "$model_dir/pytorch_model.bin/"* "$model_dir/"
            rmdir "$model_dir/pytorch_model.bin"
            echo "  ✓ Fixed!"
        else
            echo "  WARNING: No .bin files found inside"
        fi
    elif [ -f "$model_dir/pytorch_model.bin" ]; then
        echo "  ✓ Already correct (single file)"
    elif ls "$model_dir/"pytorch_model-*.bin 1> /dev/null 2>&1; then
        echo "  ✓ Already correct (sharded files in root)"
    else
        echo "  WARNING: No model files found"
    fi
    
    # Check if model.safetensors.index.json exists (for index tracking)
    if ls "$model_dir/"pytorch_model-*.bin 1> /dev/null 2>&1; then
        # Check if index file exists
        if [ ! -f "$model_dir/pytorch_model.bin.index.json" ]; then
            echo "  NOTE: Creating weight index file..."
            # Create a simple Python script to generate the index
            python3 << EOF
import os
import json
import torch

model_dir = "$model_dir"
shard_files = sorted([f for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])

if not shard_files:
    print("  No sharded files found!")
    exit(0)

weight_map = {}
total_size = 0

for shard_file in shard_files:
    shard_path = os.path.join(model_dir, shard_file)
    print(f"  Indexing {shard_file}...")
    try:
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        for key in state_dict.keys():
            weight_map[key] = shard_file
        del state_dict
    except Exception as e:
        print(f"  Error loading {shard_file}: {e}")

index = {
    "metadata": {"total_size": 0},  # We could calculate this but it's optional
    "weight_map": weight_map
}

index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
with open(index_path, "w") as f:
    json.dump(index, f, indent=2)

print(f"  ✓ Created index with {len(weight_map)} weights")
EOF
        fi
    fi
    
    echo ""
done

echo "Done! Models should now be loadable by HuggingFace/vLLM"

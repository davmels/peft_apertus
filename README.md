# Apertus Fine-Tuning Recipes

This repository provides fine-tuning recipes for Swiss AI’s Apertus language models (8B and 70B), supporting both full-parameter and LoRA-based approaches.
Built on top of popular frameworks including TRL, Accelerate, and Transformers, the recipes are optimized for efficient training on modern GPUs.
LoRA fine-tuning of the 8B model can be done on a single 40 GB GPU, while training the 70B model requires a multi-GPU setup.


## 🔗 Resources
- [Apertus 8B Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)  
- [Apertus 70B Instruct](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509)  
- [Full collection on HF](https://huggingface.co/collections/swiss-ai/apertus-llm-68b699e65415c231ace3b059)  

> 📖 For additional details about the repository, experiments, and setup validation, see [technical_report.md](technical_report.md).

---

## ⚡ Environment Setup

This project uses a containerized environment built from a Dockerfile. Follow these steps to build and register the environment.

### 1. Build the Container
Submit the build job to generate the container image (`.sqsh`):

```bash
cd dockerfile_training
sbatch build_container.sbatch
```

### 2. Configure the Environment
Once the build is complete, create a configuration file to point to the new image:

* Create a TOML file at `$HOME/.edf/<ENV_NAME>.toml` (e.g., `lsaie.toml`).
* Use the provided `lsaie.toml` as a template.
* Update the path inside the file to point to the `.sqsh` file you just built.

### 3. Launch
Start an interactive session using the environment name defined in your TOML file:

```bash
srun -A <account> --environment=lsaie --pty bash
```

---

## 📊 Dataset Configuration

The training script supports loading datasets from Hugging Face Hub or from local disk.

### Dataset Format

Your dataset should contain a `messages` column with chat-formatted data:

```python
{
    "messages": [
        {"role": "user", "content": "Your input prompt here"},
        {"role": "assistant", "content": "Expected response"}
    ]
}
```

---

## ⚙️ Training Configurations

### SFT Config Files (`configs/sft_*.yaml`)

These control model, dataset, and training hyperparameters:

| Config File | Use Case |
|-------------|----------|
| `sft_lora.yaml` | LoRA fine-tuning (memory efficient) |
| `sft_full.yaml` | Full parameter fine-tuning (8B) |
| `sft_full70.yaml` | Full parameter fine-tuning (70B) |

**Key parameters:**

```yaml
# Model
model_name_or_path: swiss-ai/Apertus-8B-Instruct-2509
dtype: bfloat16

# Dataset
dataset_name: HuggingFaceH4/Multilingual-Thinking
dataset_num_proc: 32
from_disk: false
processing_strategy: default

# Training
learning_rate: 2.0e-5
num_train_epochs: 1.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
max_length: 4096

# LoRA (only for sft_lora.yaml)
use_peft: true
lora_r: 8
lora_alpha: 32
lora_dropout: 0.0
lora_target_modules: all-linear

# Output
output_dir: ./models/my_model
report_to: "wandb"
save_steps: 100
```

---

## 🚀 DeepSpeed / Accelerate Configuration

DeepSpeed configs are in `configs/zero*.yaml`. Choose based on your memory requirements:

| Config File | ZeRO Stage | Memory Usage | Use Case |
|-------------|------------|--------------|----------|
| `zero0_multinode.yaml` | 0 | Highest | Small models, fast training |
| `zero2_multinode.yaml` | 2 | Medium | 8B models |
| `zero3_multinode.yaml` | 3 | Very Low | 70B models, multi-node |

### ZeRO Stage Explained

- **ZeRO-0**: No sharding. Replicates model, gradients, and optimizer on every GPU.
- **ZeRO-2**: Shards optimizer states and gradients across GPUs.
- **ZeRO-3**: Shards everything including model parameters.

### Choosing the Right Config

| Model Size | Recommended Config |
|------------|-------------------|
| 8B LoRA | `zero0_multinode.yaml`|
| 8B Full | `zero3_multinode.yaml` |
| 70B LoRA | `zero3_multinode.yaml` |
| 70B Full | `zero3_multinode.yaml` |

---

## 📋 SLURM Job Submission

The `slurm.sbatch` script is configured for multi-node training (Although the setup works on 1 node as well):

### Customizing the Job

Edit these variables in `slurm.sbatch` to change configs to choose the right setup for the training:

```bash
export CONFIG_FILE="$SCRATCH/LSAIE/project/peft_apertus/configs/zero3_multinode.yaml"
export SFT_CONFIG="$SCRATCH/LSAIE/project/peft_apertus/configs/sft_lora.yaml"
```

### Submit Job

```bash
sbatch slurm.sbatch
```

---

## 🎯 Interactive Fine-Tuning Examples

These examples use a single-node, multi-GPU setup (4 GPUs).

### Start Interactive Session

```bash
srun -A <account> --environment=lsaie -p normal --gpus=4 --time=02:00:00 --pty bash
```

### 8B LoRA Training

```bash
accelerate launch --num_processes=4 \
    --config_file configs/zero2_multinode.yaml \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
    --processing_strategy swiss_judgement_prediction \
    --dataset_name rcds/swiss_judgment_prediction \
    --dataset_config all \
    --output_dir ./models/apertus8B_lora
```

### 70B LoRA Training

```bash
accelerate launch --num_processes=4 \
    --config_file configs/zero3_multinode.yaml \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-70B-Instruct-2509 \
    --processing_strategy swiss_judgement_prediction \
    --dataset_name rcds/swiss_judgment_prediction \
    --dataset_config all \
    --output_dir ./models/apertus70B_lora
```

### 8B Full-Parameter Training

```bash
accelerate launch --num_processes=4 \
    --config_file configs/zero3_multinode.yaml \
    sft_train.py \
    --config configs/sft_full.yaml \
    --model_name_or_path swiss-ai/Apertus-70B-Instruct-2509 \
    --processing_strategy swiss_judgement_prediction \
    --dataset_name rcds/swiss_judgment_prediction \
    --dataset_config all \
    --output_dir ./models/apertus8B_full
```

---

## Contributors

- [Davit Melikidze](https://www.linkedin.com/in/davit-melikidze/)
- [Akmal Ashirmatov](https://www.linkedin.com/in/akmal-ashirmatov/)
- [Ben Bullinger](https://www.linkedin.com/in/benbullinger/)

## References

This work builds on the [apertus-finetuning-recipes](https://github.com/swiss-ai/apertus-finetuning-recipes) repository by [Kaustubh Ponkshe](https://kaustubhp11.github.io/) and [Raghav Singhal](https://raghavsinghal10.github.io/).
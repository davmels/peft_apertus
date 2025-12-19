FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /workspace

ENV SSL_CERT_FILE=

ENV MAX_JOBS=24 \
    CMAKE_BUILD_PARALLEL_LEVEL=24 \
    MAKEFLAGS="-j24" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (ninja-build and pdsh are required for DeepSpeed)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential cmake ninja-build pkg-config \
      libibverbs-dev libaio-dev pdsh \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# 1. FIX THE CONFLICT: Uninstall conflicting NVIDIA libs
RUN python -m pip uninstall -y cudf pylibcudf cuml cugraph || true

# 2. INSTALL PACKAGES (Added deepspeed here)
RUN python -m pip install --upgrade \
      python-dotenv \
      wandb \
      "kernels>=0.9.0" \
      "accelerate" \
      "deepspeed" \
      "git+https://github.com/huggingface/transformers.git" \
      "git+https://github.com/huggingface/peft.git" \
      "git+https://github.com/huggingface/trl.git" \
      "git+https://github.com/huggingface/datasets.git"

# 3. INSTALL XIELU
RUN python -m pip install --no-build-isolation --no-deps \
      git+https://github.com/nickjbrowning/XIELU

# 4. SANITY CHECK (Now includes deepspeed check)
RUN python - << 'PY'
import torch
import transformers, accelerate, trl, peft, datasets, deepspeed
import wandb
from dotenv import load_dotenv

print("-" * 30)
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"DeepSpeed: {deepspeed.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print("-" * 30)
PY
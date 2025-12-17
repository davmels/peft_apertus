FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

WORKDIR /workspace

# ---- System deps (INCLUDING COMPILER) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Make python visible as `python`
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip + uv
RUN python -m pip install --upgrade pip uv

# Install vLLM
RUN uv pip install --system vllm==0.12.0 --torch-backend=auto

# Common extras
RUN uv pip install --system \
    datasets \
    wandb \
    scikit-learn \
    matplotlib

# Sanity check
RUN python - << 'EOF'
import torch, vllm, shutil
print("Torch:", torch.__version__)
print("vLLM:", vllm.__version__)
print("gcc:", shutil.which("gcc"))
EOF
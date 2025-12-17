FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip uv

RUN uv pip install --system vllm==0.12.0 --torch-backend=auto
RUN uv pip install --system datasets wandb scikit-learn matplotlib

RUN python - << 'EOF'
import torch, vllm
print("Torch:", torch.__version__)
print("vLLM:", vllm.__version__)
EOF

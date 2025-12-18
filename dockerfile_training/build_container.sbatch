#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --container-writable
#SBATCH --job-name=build_sft_env
#SBATCH --output=./logs/build_%j.out

set -euo pipefail

IMAGE_NAME=sft_apertus
PROJECT_ROOT=$SCRATCH/LSAIE/project
SQSH_PATH=${PROJECT_ROOT}/containers/sft_apertus3.sqsh

mkdir -p ${PROJECT_ROOT}/containers
mkdir -p logs

echo "=== Building Podman image ==="
podman build --squash-all -t ${IMAGE_NAME}:latest .

echo "=== Removing old sqsh (if any) ==="
rm -f ${SQSH_PATH}

echo "=== Converting to SquashFS via enroot ==="
enroot import -o ${SQSH_PATH} podman://${IMAGE_NAME}:latest

echo "=== Fixing permissions ==="
setfacl -b ${SQSH_PATH}
chmod +r ${SQSH_PATH}

echo "=== Done ==="
ls -lh ${SQSH_PATH}

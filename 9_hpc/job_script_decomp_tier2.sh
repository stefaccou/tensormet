#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=decomposition_8k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


source $VSC_HOME/.bashrc
echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
module list
echo "Loaded in the environment"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 "$PROJ/metaphor/9_hpc/3_tensor_decomposition.py" --top-ks 8000

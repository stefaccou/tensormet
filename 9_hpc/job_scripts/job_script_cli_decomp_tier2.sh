#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=126000M
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --job-name=decomposition_6K_10Ki_kl
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

python3 -m tensormet.scripts.nnt \
      --dataset fineweb-en \
      --method siiSoftPlus \
      --divergence kl \
      --dim 6000 \
      --rank 100 \
      --name long \
      --random-state 1 \
      --max-cpu-frac 1 \
      --verbose t \
      --shared-factors "1-2" \
      --overwrite true \
      --iterations 10000 \
      --normalize-factors true \
      --patience 8000 \
      --return-errors full \
      --largedim true \
      --checkpoint-saving-steps 50 \
      --rec-log-every 10 \
      --sem-check-every 50 \
      --sem-error-type all \
      --resume t




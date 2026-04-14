#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=126000M
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=4gram_decomp
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
           --tier1 f \
           --dataset 4gram \
           --method siiSoftPlus \
           --divergence kl \
           --dim 4000 \
           --order 4 \
           --rank 50,50,50,50 \
           --name test_4gram \
           --shared-factors "0-1,1-2,2-3" \
           --verbose t \
           --max-cpu-frac 1 \
           --patience 1000 \
           --iterations 1000  \
           --rec-log-every 2 \
           --largedim true \
           --sem-check-every 10 \
           --checkpoint-saving-steps 25 \
           --random-state 1 \
           --normalize-factors true \
           --return-errors full \
           --overwrite true \
           --sem-fitness-target 10000




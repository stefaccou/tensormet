#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=126000M
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=frame_population
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source $VSC_HOME/.bashrc
echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
echo "Loaded in the environment"

TENSOR="$SCRATCH_DATA/tensors/frame_based"

mkdir -p "$TENSOR"

#echo "preparing data"
#cp "$DATA/vectors/fineweb_english_vectors.csv" "$VECTOR"
#echo "data prepared, submitting job"

python -m tensormet.scripts.population \
          --dataset frame_based \
          --top-ks 1000,2000,4000,6000,10000 \
          --cols-to-build frame_name,target,arg1,arg2,arg3 \
          --shared-factors 1-2,2-3,3-4
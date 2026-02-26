#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_vector_build_scp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


source "$VSC_HOME/.bashrc"
echo "Hello from $(hostname)"
echo "Loaded in the environment"

# Ensure logs dir exists (Slurm writes output there)
mkdir -p logs

VECTOR_DIR="$SCRATCH_DATA/vectors/finewebHQ_english_1B"
TENSOR_DIR="$SCRATCH_DATA/tensors/fineweb-hq"

mkdir -p "$VECTOR_DIR"
mkdir -p "$TENSOR_DIR"

# Copy everything from tier1 into VECTOR_DIR
scp -r tier1:/dodrio/scratch/projects/starting_2026_010/metaphor/data/vectors/fineweb_HQ_english_1B/* \
  "$VECTOR_DIR/"

echo "Copy complete. Files in: $VECTOR_DIR"


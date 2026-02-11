#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_vector_build
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source $VSC_HOME/.bashrc
echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
echo "Loaded in the environment"


VECTOR_DIR="$SCRATCH_DATA/vectors"
VECTOR="$VECTOR_DIR/fineweb_english_vectors.csv"
TENSOR="$SCRATCH_DATA/tensors/fineweb-en"

mkdir -p "$VECTOR_DIR"
mkdir -p "$TENSOR"

#echo "preparing data"
#cp "$DATA/vectors/fineweb_english_vectors.csv" "$VECTOR"
#echo "data prepared, submitting job"

python "$PROJ/metaphor/9_hpc/2_sparse_population.py" \
        --vectors "$VECTOR" \
        --tensors "$TENSOR"

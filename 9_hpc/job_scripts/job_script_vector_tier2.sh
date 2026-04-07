#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1 --ntasks=48
#SBATCH --mem=126000M
#SBATCH --time=04:00:00
#SBATCH --job-name=Vector_creation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source "$VSC_HOME/.bashrc"
echo "Hello from $(hostname)"
venv_cpu
echo "Loaded in the environment"

mkdir -p logs

python3 -m tensormet.scripts.vector_creation \
    --dataset fineweb \
    --output-dir "$SCRATCH_DATA/vectors/HF_fineweb" \
    --target-vectors 10000000 \
    --spacy-model en_core_web_md \
    --cpu-frac 1 \
    --log-every-s 30 \
    --batch-size 1000 \
    --rows-per-part 5000000 \
    --hf-path HuggingFaceFW/fineweb \
    --hf-config CC-MAIN-2025-26 \
    --hf-text-column text
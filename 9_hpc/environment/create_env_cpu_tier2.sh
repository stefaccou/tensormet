#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --job-name=environment_creation_cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$PROJ"

echo "starting job"
#
#module load SciPy-bundle/2024.05-gfbf-2024a
#echo "loaded scipy"
#
#module load matplotlib/3.9.2-gfbf-2024a
#echo "loaded matplotlib"

# Prefer Python 3.12 for spaCy stability.
# If your cluster exposes a specific Python 3.12 module, load it here.
# Example:
# module load Python/3.12.10-GCCcore-14.2.0


# Start clean
#rm -rf venv_cpu
#
#python3 -m venv venv_cpu --system-site-packages
#source venv_cpu/bin/activate
#echo "created and activated environment"
#
#python3 -m pip install --upgrade pip setuptools wheel
#
## CPU-specific PyTorch wheels
#python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
#
## Install core deps.
#python3 -m pip install --no-cache-dir \
#    spacy \
#    tensorflow-cpu \
#    sparse \
#    tensorly \
#    pyTensorlab \
#    tqdm \
#    pyarrow
#
#
#cd metaphor
#python3 -m pip install -e . --no-deps
module load Python/3.12.3-GCCcore-13.3.0

rm -rf venv_cpu
python3 -m venv venv_cpu
source venv_cpu/bin/activate

python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

python3 -m pip install --no-cache-dir \
    spacy \
    tensorflow-cpu \
    sparse \
    tensorly \
    pyTensorlab \
    tqdm \
    pyarrow \
    threadpoolctl \
    datasets


cd metaphor
python3 -m pip install -e . --no-deps

# Sanity checks
python3 -c "import sys; print(f'Python: {sys.version}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import numpy as np; print(f'numpy: {np.__version__}')"
python3 -c "import click, typer; print(f'click: {click.__version__} | typer: {typer.__version__}')"
python3 -c "import spacy; print(f'spaCy: {spacy.__version__}')"
python3 -c "import en_core_web_md; print('spaCy model: en_core_web_md OK')"
python3 -c "import torch; print(f'Torch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
python3 -c "from tensormet.utils import DATA_DIR, notify_discord; print(DATA_DIR); notify_discord('CPU setup completed!')"
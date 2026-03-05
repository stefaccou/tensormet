#!/bin/bash -l
#SBATCH --account=lp_tenacity
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=environment_creation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source $VSC_HOME/.bashrc
cd $PROJ

module load SciPy-bundle/2025.06-gfbf-2025a
module load matplotlib/3.10.3-gfbf-2025a
# module load tqdm/4.67.1-GCCcore-14.2.0
module load CUDA/12.6.0

python3 -m venv venv --system-site-packages
source venv/bin/activate
echo "created and activated environment"


python3 -m pip install setuptools pip --upgrade
python3 -m pip install cupy-cuda12x
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install --no-cache-dir tensorflow sparse tensorly pyTensorlab tqdm pyarrow spacy


cd metaphor
python3 -m pip install -e . --no-deps

python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import numpy as np; print(f'numpy: {np.__version__}')"
python3 -c "import cupy as cp; import torch; print('cuda:', torch.cuda.is_available())"
python3 -c "from tensormet.utils import DATA_DIR, notify_discord; print(DATA_DIR); notify_discord('setup completed!')"



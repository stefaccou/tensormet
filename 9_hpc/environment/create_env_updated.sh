#!/bin/bash -l
#PBS -N environment creation
#PBS -o $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.err
#PBS -A starting_2026_010
#PBS -l nodes=1:gpus=1
#PBS -l walltime=00:10:00
#PBS -l mem=2000mb
cd $PROJ
pwd

module load cuSPARSELt/0.6.3.2-CUDA-12.6.0 # loads in cude 12.6 and cuSPARSE
# IMPOSSIBLE
# module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 # loads in a BUNCH of stuff, with old numpy

module load SciPy-bundle/2025.06-gfbf-2025a
module load matplotlib/3.10.3-gfbf-2025a

python3 -m venv venv --system-site-packages
source venv/bin/activate
echo "created and activated environment"

export XDG_CACHE_HOME="$PROJ/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
mkdir -p "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR"

python3 -m pip install setuptools pip --upgrade
python3 -m pip install cupy-cuda12x
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install --no-cache-dir nvidia-ml-py tensorflow sparse tensorly pyTensorlab tqdm

cd metaphor
python3 -m pip install -e . --no-deps

python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "from tensormet.utils import DATA_DIR, notify_discord; print(DATA_DIR); notify_discord('environment created!')"
python3 -c "from tensormet.tucker_tensor import ExtendedTucker"


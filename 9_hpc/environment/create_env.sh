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
module load foss/2023a
module load OpenSSL/1.1

module load CuPy/13.0.0-foss-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
#module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a


module list
python3 -m venv venv-zen2 --system-site-packages
source venv-zen2/bin/activate
echo "created and activated environment"

export XDG_CACHE_HOME="$PROJ/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
mkdir -p "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR"

python3 -m pip install pip --upgrade
python3 -m pip install --no-cache-dir nvidia-ml-py sparse tensorly tqdm
python3 -m pip install pyTensorlab --no-deps

cd metaphor
python3 -m pip install -e . --no-deps

python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "from tensormet.utils import DATA_DIR; print(DATA_DIR)"
python3 -c "from tensormet.tucker_tensor import ExtendedTucker"


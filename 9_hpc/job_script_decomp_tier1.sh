#!/bin/bash -l
#PBS -N decomposition_20k
#PBS -o $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.err
#PBS -A starting_2026_010
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00


echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
module list
echo "Loaded in the environment"
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 "$PROJ/metaphor/9_hpc/3_tensor_decomposition.py" --tier1 --top-ks 20000



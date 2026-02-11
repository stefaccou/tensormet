#!/bin/bash -l
#PBS -N vector_build
#PBS -o $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.err
#PBS -A starting_2026_010
#PBS -l nodes=1
#PBS -l walltime=00:10:00




echo "Hello from $(hostname)"
# Load a CUDA toolchain

venv # custom command to activate the environment with needed dependencies.
echo "Loaded in the environment"

export INPUT_DATA=/readonly$DATA
export LOG_DIR=$PROJ/logs

echo "input = $INPUT_DATA"
echo "log = $LOG_DIR"


echo "memory available in job script"
grep -E 'MemTotal|MemAvailable' /proc/meminfo

python "$PROJ/metaphor/9_hpc/2_sparse_population.py" --top-ks 10000 20000

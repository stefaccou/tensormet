#!/bin/bash -l
#PBS -N 5gram_decomp
#PBS -o $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.err
#PBS -A starting_2026_010
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l mem=122gb
#PBS -l walltime=00:30:00

pwd
export LOG_DIR=$PROJ/logs

echo "input = $INPUT_DATA"
echo "log = $LOG_DIR"

echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
echo "Loaded in the environment"

TENSOR="$DATA/tensors/5gram"

mkdir -p "$TENSOR"

pwd
python3 -m tensormet.scripts.nnt \
           --tier1 t \
           --dataset 5gram \
           --method siiSoftPlus \
           --divergence kl \
           --dim 4000 \
           --order 5 \
           --rank 50,50,50,50,50 \
           --name testMultiDim \
           --shared-factors "0-1,1-2,2-3,3-4" \
           --verbose t \
           --max-cpu-frac 1 \
           --patience 1000 \
           --iterations 1000  \
           --rec-log-every 2 \
           --largedim true \
           --sem-check-every 10 \
           --checkpoint-saving-steps 25 \
           --random-state 1 \
           --normalize-factors true \
           --return-errors full \
           --overwrite true \
           --sem-fitness-target 10000
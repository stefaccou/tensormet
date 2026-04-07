#!/bin/bash -l
#PBS -N test_decomposition_4k
#PBS -o $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e $PROJ/logs/$PBS_JOBNAME_$PBS_JOBID.err
#PBS -A starting_2026_010
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l mem=55gb
#PBS -l walltime=02:30:00


echo "Hello from $(hostname)"
venv # custom command to activate the environment with needed dependencies.
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 "$PROJ/metaphor/9_hpc/4_config_based_launch.py" \
      --tier1 t \
      --dataset fineweb-en \
      --method siiSoftPlus \
      --divergence fr \
      --dim 6000 \
      --rank 100 \
      --name long \
      --random-state 1 \
      --max-cpu-frac 1 \
      --verbose t \
      --shared-factors "1-2" \
      --overwrite true \
      --iterations 4000 \
      --normalize-factors true \
      --patience 5000 \
      --return-errors full \
      --largedim false \
      --checkpoint-saving-steps 20 \
      --rec-log-every 5 \
      --sem-check-every 20 \
      --sem-error-type all

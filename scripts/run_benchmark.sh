#!/bin/bash

# Common sbatch variables
# PARTITION="shared-gpu,private-teodoro-gpu"
PARTITION="private-teodoro-gpu"
NUM_NODES=1
NUM_TASKS=1
TOTAL_CPU_MEMORY=64gb
NUM_CPUS_PER_TASK=8
# NUM_GPUS_PER_TASK=1

# Script variables
SIF_FOLDER=/home/users/b/borneta/sif
SIF_NAME=gemini-image.sif
SIF_IMAGE=${SIF_FOLDER}/${SIF_NAME}
SCRIPT=script.run_benchmark

# Run variables
# RUN_TYPES=("very_small")
RUN_TYPES=("small" "big")

# Function to set GPU IDs and time limit based on runtype
set_variables() {
    local runtype=$1
    case $runtype in
        "very_small")
            # GPU_IDS="gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043"
            # TIME_LIMIT="0-01:00:00"
            GPU_IDS="gpu034,gpu035"
            TIME_LIMIT="1-00:00:00"
            NUM_GPUS_PER_TASK=1
            ;;
        "small")
            # GPU_IDS="gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043"
            # TIME_LIMIT="0-10:00:00"
            GPU_IDS="gpu034,gpu035"
            TIME_LIMIT="1-12:00:00"
            NUM_GPUS_PER_TASK=1
            ;;
        "big")
            # GPU_IDS="gpu020,gpu022,gpu027,gpu028,gpu030,gpu031"
            # TIME_LIMIT="0-10:00:00"
            GPU_IDS="gpu034,gpu035"
            TIME_LIMIT="2-06:00:00"
            NUM_GPUS_PER_TASK=2
            ;;
        *)
            echo "Unknown runtype: $runtype"
            exit 1
            ;;
    esac
}

# Loop over the runtypes
for RUNTYPE in "${RUN_TYPES[@]}"
do
  set_variables "$RUNTYPE"  # to set GPU_IDS and TIME_LIMIT
  sbatch --job-name=gemini_inference_${RUNTYPE} \
         --partition=$PARTITION \
         --nodelist=$GPU_IDS \
         --nodes=$NUM_NODES \
         --ntasks=$NUM_TASKS \
         --gpus-per-task=$NUM_GPUS_PER_TASK \
         --cpus-per-task=$NUM_CPUS_PER_TASK \
         --mem=$TOTAL_CPU_MEMORY \
         --time=$TIME_LIMIT \
         --output=./results/logs/job_%j_${RUNTYPE}.txt \
         --error=./results/logs/job_%j_${RUNTYPE}.err \
         --wrap="srun apptainer exec --nv ${SIF_IMAGE} python -m ${SCRIPT} --runtype=${RUNTYPE}"
done

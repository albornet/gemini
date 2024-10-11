#!/bin/bash

# Fixed variables
PARTITION="shared-gpu,private-teodoro-gpu"
NUM_NODES=1
NUM_TASKS=1
TOTAL_CPU_MEMORY=64gb
NUM_CPUS_PER_TASK=8
NUM_GPUS_PER_TASK=1

# Dynamic variables
RUNTYPE=very_small  # very_small, small, big
if [ "$RUNTYPE" = "very_small" ]; then
    TIME_LIMIT=0-00:15:00
    GPU_IDS="gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043"
elif [ "$RUNTYPE" = "small" ]; then
    TIME_LIMIT=0-01:00:00
    GPU_IDS="gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043"
elif [ "$RUNTYPE" = "big" ]; then
    TIME_LIMIT=0-03:00:00
    GPU_IDS="gpu020,gpu022,gpu027,gpu028,gpu030,gpu031"
else
    echo "Invalid RUNTYPE value: $RUNTYPE. Must be 'small' or 'big'."
    exit 1
fi

# Script variables
SIF_FOLDER=/home/users/b/borneta/sif
SIF_NAME=gemini-image.sif
SIF_IMAGE=${SIF_FOLDER}/${SIF_NAME}

# Start an interactive session with a shell in the Apptainer container
srun --job-name=gemini_interactive_shell \
     --partition=$PARTITION \
     --nodelist=$GPU_IDS \
     --nodes=$NUM_NODES \
     --ntasks=$NUM_TASKS \
     --gpus-per-task=$NUM_GPUS_PER_TASK \
     --cpus-per-task=$NUM_CPUS_PER_TASK \
     --mem=$TOTAL_CPU_MEMORY \
     --time=$TIME_LIMIT \
     --pty apptainer shell --nv ${SIF_IMAGE}
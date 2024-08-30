#!/bin/bash

#SBATCH --job-name=gemini_inference
#SBATCH --partition=shared-gpu,private-teodoro-gpu
#SBATCH --nodelist=gpu017,gpu021,gpu025,gpu026,gpu034,gpu035,gpu044,gpu046,gpu047
#SBATCH --nodes=1  # so that only one node from the node list will be chosen
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=0-02:00:00
#SBATCH --output=./results/logs/job_%j.txt
#SBATCH --error=./results/logs/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=gemini-image.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=inference.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}

# Choosing --nodelist=???
# for 10G: gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043
# for 25G: gpu017,gpu021,gpu025,gpu026,gpu034,gpu035,gpu044,gpu046,gpu047
# for 40G: gpu020,gpu022,gpu027,gpu028,gpu030,gpu031
# for 80G: gpu029,gpu032,gpu033,gpu045
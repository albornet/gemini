#!/bin/bash

#SBATCH --job-name=gemini_inference
#SBATCH --partition=shared-gpu,private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=0-05:00:00
#SBATCH --output=/home/users/b/borneta/gemini/logs/gemini_inference/job_%j.txt
#SBATCH --error=/home/users/b/borneta/gemini/logs/gemini_inference/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=gemini-image.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=inference.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}

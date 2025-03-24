#!/bin/bash

#SBATCH --job-name=singularity_build
#SBATCH --partition=private-teodoro-gpu
#SBATCH --nodelist=gpu034,gpu035
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./results/logs/singularity_build_%j.txt
#SBATCH --error=./results/logs/singularity_build_%j.err

# Check if the .sif file exists, and only remove it if it does
if [ -f ~/sif/gemini-image.sif ]; then
    rm ~/sif/gemini-image.sif
fi

# Build the Singularity image
apptainer build ~/sif/gemini-image.sif gemini-image.def

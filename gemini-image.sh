#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=./results/logs/%x_%j.out
#SBATCH --error=./results/logs/%x_%j.err

# # Check if the .sif file exists, and only remove it if it does
# if [ -f ~/sif/gemini-image.sif ]; then
#     rm ~/sif/gemini-image.sif
# fi

# Build the Apptainer image
apptainer build ~/sif/gemini-image.sif gemini-image.def

echo "Apptainer build job finished."
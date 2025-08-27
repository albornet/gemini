#!/bin/bash

#=================================================================================
# Slurm Submitter Script for Remote Decryption on HPC
#
# Prompts for an SSH password on the login node and submits a non-interactive
# Apptainer/Singularity job to a compute node.
#=================================================================================

# Slurm job configuration
JOB_NAME=gemini-inference
PARTITION=private-teodoro-gpu
TIME=2-00:00:00
# PARTITION=shared-gpu
# TIME=0-00:10:00
MEM=64gb
NODES=1
NTASKS=1
CPUS_PER_TASK=8
GPUS_PER_TASK=1
CONSTRAINT="COMPUTE_MODEL_RTX_3090_25G|COMPUTE_MODEL_RTX_4090_25G"

# Python environment configuration
SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# Decryption script configuration
PYTHON_WRAPPER_CALL="python -m scripts.run_benchmark"
HOSTNAME="10.195.108.106"
USERNAME="borneta"
REMOTE_ENV_PATH="/home/borneta/Documents/gemini/.env"
ENCRYPTED_DATA_PATH="./data/data_2025/processed/dataset.encrypted.csv"
KEY_NAME="GEMINI"
OUTPUT_FILE="./results/decrypted_data.csv"

# Secure password prompt
echo "Please enter the SSH password for ${USERNAME}@${HOSTNAME}"
read -s -p "Password: " PASSWORD  # -s flag hides the input
echo ""  # add a newline for cleaner terminal output

# Job submission
sbatch \
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --ntasks=$NTASKS \
    --gpus-per-task=$GPUS_PER_TASK \
    --cpus-per-task=$CPUS_PER_TASK \
    --constraint=$CONSTRAINT \
    --mem=$MEM \
    --time=$TIME \
    --output=./results/logs/job_%j.txt \
    --error=./results/logs/job_%j.err \
    --wrap="srun apptainer exec --nv ${SIF_IMAGE} ${PYTHON_WRAPPER_CALL} \
        --debug \
        --encrypted-data-path \"${ENCRYPTED_DATA_PATH}\" \
        --remote-env-path \"${REMOTE_ENV_PATH}\" \
        --key-name ${KEY_NAME} \
        --hostname ${HOSTNAME} \
        --username ${USERNAME} \
        --password \"${PASSWORD}\""

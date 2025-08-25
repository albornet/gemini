#!/bin/bash

#=================================================================================
# Slurm Submitter Script for Remote Decryption on HPC
#
# Prompts for an SSH password on the login node and submits a non-interactive
# Apptainer/Singularity job to a compute node.
#=================================================================================

# Slurm job configuration
JOB_NAME="gemini-inference"
PARTITION="private-teodoro-gpu"
NUM_NODES=1
NUM_TASKS=1
TOTAL_CPU_MEMORY="64gb"
GPU_IDS="gpu034,gpu035"
TIME_LIMIT="2-00:00:00"
NUM_CPUS_PER_TASK=8
NUM_GPUS_PER_TASK=1

# Python environment configuration
SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# Decryption script configuration
PYTHON_WRAPPER_CALL="python -m scripts.run_benchmark"
HOSTNAME="10.195.108.106"
USERNAME="borneta"
REMOTE_ENV_PATH="/home/borneta/Documents/gemini/.env"
ENCRYPTED_FILE="./data/data_2025/processed/dataset.encrypted.csv"
KEY_VAR_NAME="GEMINI"
OUTPUT_FILE="./results/decrypted_data.csv"

# Secure password prompt
echo "Please enter the SSH password for ${USERNAME}@${HOSTNAME}"
read -s -p "Password: " SSH_PASSWORD  # -s flag hides the input
echo ""  # Add a newline for cleaner terminal output

# Job submission
sbatch \
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodelist=$GPU_IDS \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_TASKS \
    --gpus-per-task=$NUM_GPUS_PER_TASK \
    --cpus-per-task=$NUM_CPUS_PER_TASK \
    --mem=$TOTAL_CPU_MEMORY \
    --time=$TIME_LIMIT \
    --output=./results/logs/job_%j.txt \
    --error=./results/logs/job_%j.err \
    --wrap="srun apptainer exec --nv ${SIF_IMAGE} ${PYTHON_WRAPPER_CALL} \
        --encrypted-data-path \"${ENCRYPTED_FILE}\" \
        --remote-env-path \"${REMOTE_ENV_PATH}\" \
        --key-name ${KEY_VAR_NAME} \
        --hostname ${HOSTNAME} \
        --username ${USERNAME} \
        --password \"${SSH_PASSWORD}\""

echo "Job submitted to Slurm. Check status with 'squeue -u \$USER'."
echo "Output will be in ./results/logs/job_<jobID>.txt"

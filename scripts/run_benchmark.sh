#!/bin/bash

# --- Default Values for Arguments ---
PARTITION="private-teodoro-gpu"
TIME="0-01:00:00"
GPUS_PER_TASK=2

# --- Help/Usage Function ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --partition       Slurm partition to use        (Default: ${PARTITION})"
    echo "  -t, --time            Job time limit (D-HH:MM:SS)   (Default: ${TIME})"
    echo "  -g, --gpus-per-task   Number of GPUs per task       (Default: ${GPUS_PER_TASK})"
    echo "  -h, --help            Display this help message"
    echo "Example:"
    echo "  ./$0 -p shared-gpu -t 0-01:00:00 -g 2"
    exit 1
}

# --- Argument Parsing ---
# Loop through arguments and process them
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--partition)
        PARTITION="$2"
        shift # past argument
        shift # past value
        ;;
        -t|--time)
        TIME="$2"
        shift # past argument
        shift # past value
        ;;
        -g|--gpus-per-task)
        GPUS_PER_TASK="$2"
        shift # past argument
        shift # past value
        ;;
        -h|--help)
        usage
        ;;
        *)    # unknown option
        echo "Error: Unknown option '$1'"
        usage
        ;;
    esac
done

# --- Display Final Configuration ---
echo "Submitting job with the following configuration:"
echo "  Partition:       ${PARTITION}"
echo "  Time Limit:      ${TIME}"
echo "  GPUs per Task:   ${GPUS_PER_TASK}"
echo "-----------------------------------------------"

# Slurm job configuration (fixed values)
JOB_NAME=gemini-inference
MEM=64gb
NODES=1
NTASKS=1
CPUS_PER_TASK=8
CONSTRAINT="COMPUTE_MODEL_RTX_3090_25G|COMPUTE_MODEL_RTX_4090_25G"

# Python environment configuration
SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# Decryption script configuration
PYTHON_WRAPPER_CALL="python -m scripts.run_benchmark"
ENCRYPTED_DATA_PATH="./data/data_2025/processed/dataset.encrypted.csv"
CURATED_DATA_PATH="./data/data_2024/processed/dataset.csv"
HOSTNAME="10.195.108.106"
USERNAME="borneta"
REMOTE_ENV_PATH="/home/borneta/Documents/gemini/.env"
KEY_NAME="GEMINI"
OUTPUT_FILE="./results/decrypted_data.csv"

# Password prompt
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
        --encrypted-data-path \"${ENCRYPTED_DATA_PATH}\" \
        --curated-data-path \"${CURATED_DATA_PATH}\" \
        --remote-env-path \"${REMOTE_ENV_PATH}\" \
        --key-name ${KEY_NAME} \
        --hostname ${HOSTNAME} \
        --username ${USERNAME} \
        --password \"${PASSWORD}\""

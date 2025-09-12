#!/bin/bash

# --- Default Values for Arguments ---
PARTITION="private-teodoro-gpu"
TIME="0-00:10:00"
GPUS_PER_TASK=2
w
# --- Help/Usage Function ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --partition       Slurm partition to use       (Default: ${PARTITION})"
    echo "  -t, --time            Job time limit (D-HH:MM:SS)  (Default: ${TIME})"
    echo "  -g, --gpus-per-task   Number of GPUs per task      (Default: ${GPUS_PER_TASK})"
    echo "  -h, --help            Display this help message"
    echo "Example:"
    echo "  ./$0 -p shared-gpu -t 0-01:00:00 -g 2"
    exit 1
}

# --- Argument Parsing (Robust version) ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--partition)
        if [[ -n "$2" && "$2" != -* ]]; then
            PARTITION="$2"
            shift # past argument
            shift # past value
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -t|--time)
        if [[ -n "$2" && "$2" != -* ]]; then
            TIME="$2"
            shift # past argument
            shift # past value
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
        ;;
        -g|--gpus-per-task)
        if [[ -n "$2" && "$2" != -* ]]; then
            GPUS_PER_TASK="$2"
            shift # past argument
            shift # past value
        else
            echo "Error: Argument for $1 is missing" >&2
            exit 1
        fi
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

# --- Slurm job configuration (fixed values) ---
JOB_NAME=gemini-inference
MEM=64gb
NODES=1
NTASKS=1
CPUS_PER_TASK=8
CONSTRAINT="COMPUTE_MODEL_RTX_3090_25G|COMPUTE_MODEL_RTX_4090_25G"

# --- Python environment configuration ---
SIF_FOLDER="/home/users/b/borneta/sif"
SIF_NAME="gemini-image.sif"
SIF_IMAGE="${SIF_FOLDER}/${SIF_NAME}"

# --- Decryption script configuration ---
PYTHON_WRAPPER_CALL="python -m scripts.run_benchmark"
ENCRYPTED_DATA_PATH="./data/data_2025/processed/dataset.encrypted.csv"
CURATED_DATA_PATH="./data/data_2024/processed/dataset.csv"
HOSTNAME="10.195.108.106"
USERNAME="borneta"
REMOTE_ENV_PATH="/home/borneta/Documents/gemini/.env"
KEY_NAME="GEMINI"
OUTPUT_FILE="./results/decrypted_data.csv"

# Job submission uses a "here document"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --gpus-per-task=${GPUS_PER_TASK}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --constraint="${CONSTRAINT}"
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=./results/logs/job_%j.txt
#SBATCH --error=./results/logs/job_%j.err

# --- Job Execution ---
echo "Starting job \$SLURM_JOB_ID on node \$(hostname)..."

# The srun command to be executed on the compute node.
# Note that the --password argument is no longer needed.
srun apptainer exec --nv "${SIF_IMAGE}" ${PYTHON_WRAPPER_CALL} \\
    --encrypted-data-path "${ENCRYPTED_DATA_PATH}" \\
    --curated-data-path "${CURATED_DATA_PATH}" \\
    --remote-env-path "${REMOTE_ENV_PATH}" \\
    --key-name "${KEY_NAME}" \\
    --hostname "${HOSTNAME}" \\
    --username "${USERNAME}"

echo "Job finished with exit code \$?."

EOF

# Let the user know the job has been submitted
echo "Job has been submitted to Slurm."
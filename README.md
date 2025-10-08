# Gemini project

This project is designed for large-scale machine learning experiments, particularly focusing on Large Language Models (LLMs) using `vllm`.

The project is organized as follows:

```
.
├── configs/              # Configuration files (e.g., for models, experiments)
├── data/                 # Raw and processed data (ignored by git)
├── experiments/          # Experiment-specific scripts or notebooks
├── results/              # Output files, logs, and saved models (ignored by git)
├── scripts/              # Helper and automation scripts
│   └── build_image.sh    # Script to build the Apptainer image on HPC
├── src/                  # Main source code for the project
├── .gitignore            # Specifies intentionally untracked files to ignore
├── gemini-image.def      # Apptainer/Singularity definition file
├── README.md             # This file
└── requirements.txt      # Python package requirements
```

## Setup and installation

You can set up the environment locally for development or build a container for reproducible execution on an HPC cluster.

### Local development setup

For local development, we recommend using `uv` for fast dependency management. `uv` is a very fast Python package installer and resolver, written in Rust.

1.  **Install `uv`**

    If you don't have `uv` installed, you can install it with:
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    For other installation methods, see the uv installation guide.

2.  **Create a virtual environment and install dependencies**

    From the root of the project, run the following commands:
    ```bash
    # Create a virtual environment named .venv
    uv venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Install the required packages
    uv pip install -r requirements.txt
    ```

### HPC with Apptainer setup

For running experiments on an HPC cluster, an Apptainer (formerly Singularity) container is used to ensure a consistent and reproducible environment.

1.  **Prerequisites**

    -   Access to an HPC cluster with Apptainer/Singularity installed.
    -   A SLURM workload manager (the build script is written for SLURM).

2.  **Build the Container image**

    The `gemini-image.def` file defines the container environment. It uses a PyTorch base image, sets up a virtual environment with `uv`, and installs all necessary packages from `requirements.txt`.

    To build the image, submit the `gemini-image.sh` script to the SLURM scheduler:

    ```bash
    sbatch scripts/build_image.sh
    ```

    This script will:
    -   Request resources on the cluster.
    -   Build the Apptainer image `~/sif/gemini-image.sif` from `gemini-image.def`.
    -   Store build logs in `results/logs/`.

    You can monitor the build job with `squeue -u $USER`.

## Usage

Once the setup is complete, you can run your experiments.

### Running experiments locally

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
./experiments/experiment_1_desktop.sh  # after checking the parameters in this file
```

### Running experiments on HPC

To run the experiment from an HPC, use the following script:
```bash
./experiments/experiment_1.sh  # after checking the parameters in this file
```

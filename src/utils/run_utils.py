import os
import torch
import yaml
from argparse import ArgumentParser


def add_model_arguments(parser: ArgumentParser) -> None:
    """ Parse and validate model arguments
    """
    model_group = parser.add_argument_group(
        title="Model configuration",
        description="Configuration options for model benchmarking",
    )

    model_group.add_argument(
        "-mc", "--model-config-path",
        default="configs/model_config.yaml",
        help="Path to the model configuration file"
    )

    model_group.add_argument(
        "-pc", "--prompt-config-path",
        default="configs/prompt_config.yaml",
        help="Path to the prompt configuration file"
    )

    model_group.add_argument(
        "-oc", "--output-config-path",
        default="configs/output_config.yaml",
        help="Path to the output schema configuration file"
    )


def add_data_arguments(parser: ArgumentParser) -> None:
    """
    Add arguments required for reading an encrypted file by fetching a remote key.
    """
    data_group = parser.add_argument_group(
        title="Data access and remote env configuration",
        description="Arguments for the local encrypted file and key identifier."
    )

    data_group.add_argument(
        "--encrypted-data-path",
        "-ed",
        type=str,
        required=True,
        help="Path to the local encrypted data file.",
    )

    data_group.add_argument(
        "--key-name",
        "-kn",
        type=str,
        required=True,
        help="Name of the encryption key variable in the .env file on the remote server.",
    )
    data_group.add_argument(
        "--hostname",
        "-hn",
        type=str,
        required=True,
        help="Hostname or IP address of the remote server.",
    )

    data_group.add_argument(
        "--username",
        "-un",
        type=str,
        required=True,
        help="Username for the SSH connection.",
    )

    data_group.add_argument(
        "--remote-env-path",
        "-re",
        type=str,
        required=True,
        help="Path to the .env file on the remote server.",
    )

    data_group.add_argument(
        "--port",
        type=int,
        default=22,
        help="SSH port on the remote server (default: 22).",
    )

    data_group.add_argument(
        "--private-key-path",
        type=str,
        default=None,
        help="Path to the private SSH key for authentication (optional).",
    )

    data_group.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for SSH authentication, if no key is used (optional).",
    )


def _load_config_from_yaml(config_file_path: str) -> dict:
    """ Load configuration from a YAML file
    
    Args:
        config_file_path (str): Path to the YAML
    """
    try:
        with open(config_file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file_path}")
        exit(1)


def load_config_files(script_args) -> dict:
    """ Load configurations from YAML files specified in script_args
    """
    # Load configurations
    model_config = _load_config_from_yaml(script_args.model_config_path)
    prompt_config = _load_config_from_yaml(script_args.prompt_config_path)
    output_config = _load_config_from_yaml(script_args.output_config_path)

    # Combine all configurations into a single dictionary
    run_config = {**model_config, **prompt_config, **output_config}

    return run_config


def set_torch_cuda_arch_list() -> None:
    """
    Sets the TORCH_CUDA_ARCH_LIST environment variable to the CUDA compute
    capability of the current GPU(s). This prevents PyTorch from compiling
    kernels for all possible architectures, reducing compilation time.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping TORCH_CUDA_ARCH_LIST setup.")
        return

    # Get the current GPU's compute capability.
    # For example, on an RTX 3080, this will return a tuple like (8, 6).
    major, minor = torch.cuda.get_device_capability(device=None)
    
    # Note: the required format is 'X.Y', not 'sm_XY'.
    cuda_arch_value = f"{major}.{minor}"
    
    # Check if the environment variable is already set correctly
    if os.environ.get('TORCH_CUDA_ARCH_LIST') != cuda_arch_value:
        print(f"Setting TORCH_CUDA_ARCH_LIST to '{cuda_arch_value}' for efficient compilation.")
        os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch_value
    else:
        print(f"TORCH_CUDA_ARCH_LIST is already set to '{cuda_arch_value}'.")


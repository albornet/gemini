import yaml
from itertools import chain
from argparse import ArgumentParser, Namespace


def _parse_script_args() -> Namespace:
    """ Parse and validate command line arguments
    
    Returns:
        Namespace containing:
        - runtype: str (very_small, small, big, all)
        - debug: bool
        - plot_only: bool
    """
    parser = ArgumentParser(description="Benchmark LLMs on healthcare tasks")
    
    # Run type configuration
    parser.add_argument(
        "-t", "--runtype",
        default="very_small",
        choices=["very_small", "small", "large", "all"],
        help="Scope of benchmark: very_small, small, big, or all"
    )

    # General configuration
    parser.add_argument(
        "-r", "--run_config_path",
        default="configs/run_params.yaml",
        help="Path to the run configuration file"
    )

    # Specific configuration for which model will be run
    parser.add_argument(
        "-m", "--model_config_path",
        required=True,
        help="Path to the model configuration file"
    )
    
    # Debug mode flag
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    # Plot mode flag
    parser.add_argument(
        "-p", "--plot_only",
        action="store_true",
        help="Enable plot mode (to re-plot saved results)"
    )

    return parser.parse_args()


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


def _convert_dict_to_namespace(data: dict) -> Namespace:
    """ Recursively convert a dictionary and its nested dictionaries into
        argparse.Namespace objects
    """
    if not isinstance(data, dict):
        return data

    namespace_obj = Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(namespace_obj, key, _convert_dict_to_namespace(value))
        else:
            setattr(namespace_obj, key, value)
    return namespace_obj


def load_config() -> Namespace:
    """ Load configurations for benchmarking models into a single object
    """
    # Load configurations
    script_args = _parse_script_args()
    run_config = _load_config_from_yaml(script_args.run_config_path)
    model_config = _load_config_from_yaml(script_args.model_config_path)

    # Merge configurations into a single object
    cfg = {**run_config, **model_config}
    cfg["RUNTYPE"] = script_args.runtype.lower()
    cfg["DEBUG"] = script_args.debug
    cfg["PLOT_ONLY"] = script_args.plot_only

    # Select which models are going to be run
    if cfg["RUNTYPE"] == "all":
        models_to_benchmark = list(chain(*cfg["models"]))
    else:
        models_to_benchmark = cfg["models"][cfg["RUNTYPE"]]
    cfg["MODELS_TO_BENCHMARK"] = models_to_benchmark

    # Return a namespace object
    return _convert_dict_to_namespace(cfg)
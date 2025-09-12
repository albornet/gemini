import os
import gc
import argparse
import pandas as pd
from typing import Any
from functools import partial
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from vllm import LLM
from datasets import Dataset
from src.models.llm_evaluation import (
    get_gpu_memory_usage_by_pid,
    compute_and_save_metrics,
    plot_metrics,
    print_gpu_info,
)
from src.models.llm_inference import process_samples
from src.models.llm_loader import get_model_and_tokenizer
from src.data.prompting import build_prompt
from src.data.encryption import read_pandas_from_encrypted_file
from src.utils.run_utils import (
    load_config_files,
    add_model_arguments,
    add_data_arguments,
    set_torch_cuda_arch_list,
)


def main(args: argparse.Namespace):
    """
    Run benchmarks on different generative large language models in separate
    processes to avoid GPU memory leak or accumulation
    """
    # Force a fresh run for each new process
    torch_mp.set_start_method("spawn", force=True)

    # Load configuration from yaml files
    cfg = load_config_files(args)

    # Load benchmarking dataset (only if a benchmark is actually run)
    dataset = None
    if not args.plot_only:
        data_loading_args = cfg["data_loading_arguments"]
        dataset = load_data_formatted_for_benchmarking(args, **data_loading_args)

    # TODO: LOOP THAT MODIFIES RUN_KWARGS FOR EACH MODEL TO BE BENCHMARKED (?)
    run_kwargs = {"cfg": cfg, "dataset": dataset, "debug": args.debug}
    if args.debug:
        record_one_benchmark(**run_kwargs)
    else:
        process = torch_mp.Process(target=record_one_benchmark, kwargs=run_kwargs)
        process.start()  # spawn a new process for each benchmark run
        process.join()  # wait for the process to complete before continuing


def load_data_formatted_for_benchmarking(
    args: argparse.Namespace,
    use_curated_dataset: bool = False,
    remove_samples_without_label: bool = False,
    sample_small_dataset: bool = False,
    min_samples_per_class: int = 200,
) -> Dataset:
    """
    Load and preprocess data for benchmarking
    """
    # Load dataset file (small, curated dataset)
    if use_curated_dataset:
        df_data = pd.read_csv(args.curated_data_path)

    # Load dataset file (large, non-curated dataset + encrypted)
    else:
        print("Loading encrypted dataset...")
        df_data = read_pandas_from_encrypted_file(
            encrypted_file_path=args.encrypted_data_path,
            encryption_key_var_name=args.key_name,
            hostname=args.hostname,
            username=args.username,
            remote_env_path=args.remote_env_path,
            port=args.port,
        )

    # Check for the presence of benchmarking fields
    if "input_text" not in df_data.columns or "label" not in df_data.columns:
        raise KeyError("Missing expected columns: 'input_text', 'label'")

    # Replace label values and / or filter out samples without labels if specified
    if remove_samples_without_label:
        print("Filtering out samples without labels.")
        df_data = df_data.dropna(subset=["label"])

    # Benchmark the chosen model
    if sample_small_dataset:
        df_data = sample_small_balanced_dataset(df_data, min_samples_per_class)
    dataset = Dataset.from_pandas(df_data)
    
    return dataset


def sample_small_balanced_dataset(
    df_data: pd.DataFrame,
    min_samples_per_class: int = 200,
) -> pd.DataFrame:
    """
    Select a small portion of the data that has more or less balanced classes
    """
    print("Sampling a small, balanced dataset for debugging.")
    df_data = df_data.groupby("label", group_keys=False)
    df_data = df_data.apply(
        lambda x: x.sample(n=min(len(x), min_samples_per_class)), 
        include_groups=True,
    )
    df_data = df_data.sample(frac=1)
    df_data = df_data.reset_index(drop=True)
    
    return df_data


def record_one_benchmark(
    cfg: dict,
    dataset: Dataset,
    debug: bool = False,
) -> None:
    """ Run the benchmark for a single model and record metrics
    """
    # If not is dataset provided, skip benchmarking and re-plot existing results
    if dataset is None:
        benchmark_results = None

    # Run something only if not in plot_only mode
    else:
        print(f"Benchmarking {cfg['model_path']} with {cfg['inference_backend']} backend")
        print_gpu_info()

        # Initialize model with the required backend
        try:
            model, tokenizer = get_model_and_tokenizer(**cfg, debug=debug)
        except ValueError as e:
            if cfg["skip_to_next_model_if_error"]:
                print(f"The following exception occurred: {e}. Skipping to next model.")
            raise  # this will return None

        # Build prompts using the model tokenizer
        # dataset = dataset.copy()  # to avoid modifying the original dataset
        dataset = dataset.map(
            function=partial(build_prompt, cfg=cfg, tokenizer=tokenizer),
            desc="Building prompts",
        )

        # Record results obtained with the selected model
        benchmark_results = benchmark_one_model(
            cfg=cfg,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
        )

    # Compute and save benchmark results
    print("Benchmarked %s" % cfg["model_path"])
    save_benchmark_results(cfg=cfg, benchmark_results=benchmark_results)

    # Clean memory for the next benchmark
    if "tokenizer" in locals(): del tokenizer
    if "model" in locals(): del model
    if torch_dist.is_initialized(): torch_dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()


def save_benchmark_results(
    cfg: dict,
    benchmark_results: dict,
):
    """ Save benchmark results to a CSV file and plot metrics
    """
    # Build unique output path given configuration
    output_subdir = cfg["inference_backend"]
    if cfg["use_output_guide"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["result_dir"], output_subdir)
    if cfg["quant_method"] is not None and cfg["quant_method"].lower() == "gguf":
        model_result_path = f"{cfg['model_path']}-{cfg['quant_scheme']}.csv"
    else:
        model_result_path = f"{cfg['model_path']}-no_quant_scheme.csv"
    output_path = os.path.join(output_dir, model_result_path)
    
    # If provided, save model results to a csv file
    if benchmark_results is not None:
        compute_and_save_metrics(
            benchmark_results=benchmark_results,
            model_path=cfg["model_path"],
            output_path=output_path,
        )

    # Metrics are plotted by loading saved benchmark results
    metric_path = output_path.replace(".csv", ".json")
    plot_metrics(metric_path=metric_path)


def benchmark_one_model(
    cfg: dict[str, Any],
    dataset: Dataset,
    model: AutoModelForCausalLM|LLM|Llama,
    tokenizer: AutoTokenizer,
) -> dict[str, Dataset|float]:
    """
    Prompt a generative LLM with medical questions and computes metrics
    about computation time and GPU memory usage
    https://triton-lang.org/main/python-api/generated/triton.testing.do_bench

    Args:
        cfg (dict): inference configuration parameters
        model (AutoModelForCausalLM): actual LLM doing inference in the benchmark   
        tokenizer (AutoTokenizer): tokenizer used by the LLM

    Returns:
        Dataset: model outputs, with computation time and GPU memory usage
    """
    # Initialize events, clear cache and reset peak memory stats
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Time execution time for processing samples
    start_event.record()
    dataset_with_outputs = process_samples(dataset, model, tokenizer, **cfg)
    end_event.record()

    # Record the time and memory usage
    time = start_event.elapsed_time(end_event) / 1000  # in seconds
    time = time / len(dataset)  # time per sample
    # time = time / cfg["n_inference_repeats"]  # time per inference (?)
    memory = get_gpu_memory_usage_by_pid()  # in GB

    # Return benchmark results for metric computation and plotting
    print("Model successfully benchmarked.")
    return {"dataset": dataset_with_outputs, "time": time, "memory": memory}


if __name__ == "__main__":
    # Provide an architecture list to help the inference backends
    set_torch_cuda_arch_list()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM in inference mode for data extraction."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Run the script in debug mode with a smaller dataset and fewer repetitions."
    )
    parser.add_argument(
        "--plot-only", "-po", action="store_true",
        help="Only plot the results from a previouly run benchmark."
    )
    add_model_arguments(parser)
    add_data_arguments(parser)
    args = parser.parse_args()

    # Start the main script
    main(args)

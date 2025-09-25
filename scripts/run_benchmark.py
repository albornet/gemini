import os
import gc
import argparse
from typing import Any
from functools import partial
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoModelForCausalLM
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
from src.models.llm_loader import load_model
from src.data.prompting import build_messages
from src.data.data_loading import load_data_formatted_for_benchmarking
from src.utils.run_utils import (
    load_config_files,
    add_model_arguments,
    add_data_arguments,
    set_torch_cuda_arch_list,
    extract_quant_method,
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

    # Simply re-plot previous results if specified
    if args.plot_only:
        save_benchmark_results(cfg=cfg)

    # Load data and run the benchmark
    else:
        data_loading_args = cfg["data_loading_arguments"]
        dataset = load_data_formatted_for_benchmarking(args, **data_loading_args)
        run_kwargs = {"cfg": cfg, "dataset": dataset, "debug": args.debug}
        record_one_benchmark(**run_kwargs)


def record_one_benchmark(
    cfg: dict,
    dataset: Dataset,
    debug: bool = False,
) -> None:
    """ Run the benchmark for a single model and record metrics
    """
    # Try to benchmark the model
    model, server_process = None, None
    try:

        # Model loading (and server start if needed)
        print(f"Benchmarking {cfg['model_path']} with {cfg['inference_backend']} backend")
        print_gpu_info()
        model, server_process = load_model(**cfg, debug=debug)
        
        # Build messages using the model configuration
        dataset = dataset.map(
            function=partial(build_messages, cfg=cfg),
            desc="Building messages for prompting",
        )

        # Record results
        benchmark_results = benchmark_one_model(
            cfg=cfg,
            dataset=dataset,
            model=model,
        )

        # Compute and save benchmark results
        save_benchmark_results(cfg=cfg, benchmark_results=benchmark_results)
        print("Benchmarked %s" % cfg["model_path"])

    # Error handling logic
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        raise 
    
    # Cleanup Logic
    finally:
        print("Cleaning up resources...")

        # Terminate server process if it was started
        if server_process is not None:
            print("Terminating vLLM server...")
            server_process.terminate()
            server_process.wait(timeout=10) # Wait for process to exit
            if server_process.poll() is None:
                print("Server did not terminate gracefully, killing.")
                server_process.kill()
            print("Server terminated.")

        # Clean GPU memory and distributed processes if any
        if "model" in locals() and cfg.get('inference_backend') != 'vllm-serve':
            del model
        if torch_dist.is_initialized(): torch_dist.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_info()
        print("Cleaned memory")


def benchmark_one_model(
    cfg: dict[str, Any],
    dataset: Dataset,
    model: AutoModelForCausalLM | LLM | Llama,
) -> dict[str, Dataset|float]:
    """
    Prompts a generative LLM with medical questions and computes metrics
    about computation time and GPU memory usage
    --> https://triton-lang.org/main/python-api/generated/triton.testing.do_bench

    """
    # Initialize events, clear cache and reset peak memory stats
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Time execution time for processing samples
    start_event.record()
    dataset_with_outputs = process_samples(dataset, model, **cfg)
    end_event.record()

    # Wait for all kernels on all GPUs to finish BEFORE measuring time.
    torch.cuda.synchronize()
    print("Devices synchronized.")

    # Record the time and memory usage
    time = start_event.elapsed_time(end_event) / 1000  # in seconds
    time = time / len(dataset)  # time per sample
    # time = time / cfg["n_inference_repeats"]  # time per inference (?)
    memory = get_gpu_memory_usage_by_pid()  # in GB

    # Return benchmark results for metric computation and plotting
    print("Model successfully benchmarked.")
    return {"dataset": dataset_with_outputs, "time": time, "memory": memory}


def save_benchmark_results(
    cfg: dict,
    benchmark_results: dict | None = None,
) -> None:
    """
    Saves benchmark results to a CSV file and plot metrics
    """
    # Build unique output path given configuration
    output_subdir = cfg["inference_backend"]
    if cfg["use_output_guide"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["result_dir"], output_subdir)
    quant_method = extract_quant_method(cfg["model_path"])
    if quant_method is not None and quant_method == "gguf":
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

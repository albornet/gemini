import os
import gc
import argparse    
from typing import Any
from functools import partial
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from vllm import LLM
from datasets import Dataset
from src.models.llm_benchmark import (
    do_bench,
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
)


def main(args):
    """ Run benchmarks on different generative large language models in separate
        processes to avoid GPU memory leak or accumulation
    """
    print("DEBUG WARNING REMOVE THE 0.4 IN VLLM INIT")
    print("DEBUG WARNING REMOVE THE 0.4 IN VLLM INIT")
    print("DEBUG WARNING REMOVE THE 0.4 IN VLLM INIT")
    print("DEBUG WARNING REMOVE THE 0.4 IN VLLM INIT")
    print("DEBUG WARNING REMOVE THE 1 IN MODEL CONFIG")
    print("DEBUG WARNING REMOVE THE 1 IN MODEL CONFIG")
    print("DEBUG WARNING REMOVE THE 1 IN MODEL CONFIG")
    print("DEBUG WARNING REMOVE THE 1 IN MODEL CONFIG")
    # Force a fresh run for each new process
    torch_mp.set_start_method("spawn", force=True)

    # Load configurations from YAML files
    cfg = load_config_files(args)

    # Load dataset from encrypted file
    print("Loading encrypted dataset...")
    df_data = read_pandas_from_encrypted_file(
        encrypted_file_path=args.encrypted_data_path,
        encryption_key_var_name=args.key_name,
        hostname=args.hostname,
        username=args.username,
        remote_env_path=args.remote_env_path,
        port=args.port,
        password=args.password,
        private_key_path=args.private_key_path,
    )

    # Post-process the data for our specific case (TODO: DO IT MORE GENERAL?)
    input_label_map = {"Anonymised letter": "input_text", "Label_student": "label"}
    df_data.rename(columns=input_label_map, inplace=True)
    df_data = df_data.dropna(subset="label")
    df_data = df_data[~df_data["label"].isin(["No FU", "No FU yet"])]

    # Benchmark the chosen model
    dataset = Dataset.from_pandas(df_data)
    if args.debug: dataset = Dataset.from_dict(mapping=dataset[:5])
    run_kwargs = {"cfg": cfg, "dataset": dataset, "debug": args.debug}

    # TODO: LOOP THAT MODIFIES RUN_KWARGS FOR EACH MODEL TO BE BENCHMARKED
    if args.debug:
        record_one_benchmark(**run_kwargs)
    else:
        process = torch_mp.Process(target=record_one_benchmark, kwargs=run_kwargs)
        process.start()  # spawn a new process for each benchmark run
        process.join()   # wait for the process to complete before continuing

    # Success message
    print("\nScript completed successfully!")


def record_one_benchmark(
    cfg: dict,
    dataset: Dataset,
    debug: bool=False,
) -> None:
    """ Runs the benchmark for a single model and record metrics
    """
    print_gpu_info()
    print(f"Benchmarking {cfg['model_path']} with {cfg['inference_backend']} backend")

    # Initialize model with the required backend
    try:
        model, tokenizer = get_model_and_tokenizer(**cfg)
    except ValueError as e:
        if cfg["skip_to_next_model_if_error"]:
            print(f"The following exception occurred: {e}. Skipping to next model.")
        raise(e)  # this will return None

    # Build prompts using the model tokenizer
    # dataset = dataset.copy()  # to avoid modifying the original dataset
    dataset = dataset.map(
        function=partial(build_prompt, cfg=cfg, tokenizer=tokenizer),
        desc="Building prompts",
    )

    # Record results obtained with the selected model
    n_inference_repeats = 2 if debug else cfg["n_inference_repeats"]
    benchmark_results = benchmark_one_model(
        cfg=cfg,
        dataset=dataset,
        model=model,
        model_path=cfg["model_path"],
        tokenizer=tokenizer,
        n_inference_repeats=n_inference_repeats,
    )

    # Compute and save benchmark results
    print("Benchmarked %s" % cfg["model_path"])
    save_benchmark_results(cfg=cfg, benchmark_results=benchmark_results)

    # Clean memory for the next benchmark
    if "tokenizer" in locals(): del tokenizer
    if "model" in locals(): del model
    if torch_dist.is_initialized(): torch_dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def save_benchmark_results(cfg, benchmark_results):
    """ Save benchmark results to a CSV file and plot metrics
    """
    # Build unique output path given configuration
    output_subdir = cfg["inference_backend"]
    if cfg["use_output_guide"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["result_dir"], output_subdir)
    model_result_path = f"{cfg['model_path']}-{cfg['quant_scheme']}.csv"
    output_path = os.path.join(output_dir, model_result_path)
    
    # Save model results to a csv file
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
    model_path: str,
    tokenizer: AutoTokenizer,
    n_inference_repeats: int,
) -> dict:
    """
    Prompt a large language model with medical questions and computes metrics
    about computation time, GPU memory usage, and error rate

    Args:
        model (AutoModelForCausalLM): actual LLM doing inference in the benchmark   
        tokenizer (AutoTokenizer): tokenizer used by the LLM
        model_path (str): path to identify and save the model

    Returns:
        dict: benchmark metrics including time and memory usage
    """
    # Extract data while monitoring computation time and GPU memory usage
    bench_fn = lambda: process_samples(dataset, model, tokenizer, **cfg)
    outputs, times, memories = do_bench(
        bench_fn=bench_fn,
        model_path=model_path,
        n_repeats=n_inference_repeats,
        return_outputs=True,
    )
    times = times / len(dataset)  # since we want time per sample

    # Combine all outputs and add them to the input dataset
    common_cols = set(dataset.column_names)
    for i, output_ds in enumerate(outputs):
        for col in output_ds.column_names:
            if col not in common_cols:
                dataset = dataset.add_column(f"{col}_{i:03d}", output_ds[col])

    # Return benchmark results for metric computation and plotting
    return {"dataset": dataset, "times": times, "memories": memories}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM in inference mode for data extraction."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Run the script in debug mode with a smaller dataset and fewer repetitions."
    )
    add_model_arguments(parser)
    add_data_arguments(parser)
    args = parser.parse_args()
    main(args)


# python -m scripts.run_benchmark \
#     -ed "./data/data_2025/processed/dataset.encrypted.csv" \
#     -re "/home/users/b/borneta/gemini/.env" \
#     -hn "login1.baobab.hpc.unige.ch" \
#     -un borneta \
#     -kn GEMINI
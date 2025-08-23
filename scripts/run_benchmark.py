import os
import gc
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from vllm import LLM
from datasets import Dataset
from functools import partial
from src.models.llm_benchmark import do_bench, compute_and_save_metrics, plot_metrics, print_gpu_info
from src.models.llm_inference import process_samples
from src.models.llm_loader import get_model_and_tokenizer
from src.utils.run_utils import load_config
from src.data.prompting import build_prompt
from src.data.encryption import read_pandas_from_encrypted_file

cfg = load_config()


def main(args):
    """ Run benchmarks on different generative large language models in separate
        processes to avoid GPU memory leak or accumulation
    """
    # Force a fresh run for each new process
    torch_mp.set_start_method("spawn", force=True)

    # Load data
    df_data = read_pandas_from_encrypted_file(
        encrypted_file_path=cfg["ENCRYPTED_DATASET_PATH"],
        encryption_key_var_name=cfg["ENCRYPTION_KEY_VAR_NAME"],
    )
    dataset = Dataset.from_pandas(df_data)
    if cfg["DEBUG"]: dataset = Dataset.from_dict(mapping=dataset[:5])

    # Benchmark all models sequentially, spawning one process per benchmark
    print(f"Processing {cfg['RUNTYPE']} models\n")
    for model_info in cfg["MODELS_TO_BENCHMARK"]:
        if cfg["FIGURES_ONLY"] or cfg["DEBUG"]:
            record_one_benchmark(**model_info)
        else:
            process = torch_mp.Process(target=record_one_benchmark, kwargs=model_info)
            process.start()  # spawn a new process for each benchmark run
            process.join()   # wait for the process to complete before continuing

    # Success message
    print("\nScript completed successfully!")


def record_one_benchmark(
    dataset: Dataset,
    model_path: str,
    quant_method: str,
    quant_scheme: str="no-quant-scheme",
) -> None:
    """ Runs the benchmark for a single model and record metrics
    """
    # Build unique output path given configuration
    output_subdir = cfg["INFERENCE_BACKEND"]
    if cfg["USE_OUTPUT_GUIDE"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["RESULT_DIR"], output_subdir)
    output_path = os.path.join(output_dir, f"{model_path}-{quant_scheme}.csv")

    # Do not run any inference if already generated data is just being replotted
    if cfg["FIGURES_ONLY"]:
        print("Replotting data for %s" % model_path)

    # Actual benchmark run
    else:
        print_gpu_info()
        print(f"Benchmarking {model_path} with {cfg['INFERENCE_BACKEND']} backend")

        # Initialize model with the required backend
        try:
            # model, tokenizer = get_model_and_tokenizer(
            #     model_path=model_path,
            #     quant_method=quant_method,
            #     quant_scheme=quant_scheme,
            # )
            model, tokenizer = None, None  # DEBUG FOR DATA ENCRYPTION
        except ValueError as e:
            if cfg["SKIP_TO_NEXT_MODEL_IF_ERROR"]:
                print(f"The following exception occurred: {e}. Skipping to next model.")
                raise(e)  # return None
            else:
                raise e

        # Record results obtained with the selected model
        benchmark_results = benchmark_one_model(
            model=model,
            model_path=model_path,
            tokenizer=tokenizer,
        )
        compute_and_save_metrics(
            benchmark_results=benchmark_results,
            model_path=model_path,
            output_path=output_path,
        )
        print("Benchmarked %s" % model_path)

    # Metrics are plotted by loading saved benchmark results
    metric_path = output_path.replace(".csv", ".json")
    plot_metrics(metric_path=metric_path)

    # Clean memory for the next benchmark
    if "tokenizer" in locals(): del tokenizer
    if "model" in locals(): del model
    if torch_dist.is_initialized(): torch_dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_one_model(
    model: AutoModelForCausalLM|LLM|Llama,
    model_path: str,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
) -> dict:
    """ Prompt a large language model with medical questions and computes
        metrics about computation time, GPU memory usage, and error rate
    
    Args:
        model (AutoModelForCausalLM): actual LLM doing inference in the benchmark   
        tokenizer (AutoTokenizer): tokenizer used by the LLM
        model_path (str): path to identify and save the model
        
    Returns:
        dict: benchmark metrics including time and memory usage
    """


    # Build prompted dataset using the model tokenizeravoid modifying the original one
    dataset = dataset.copy()
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), desc="Building prompts")
    
    # Measure computation time and GPU memory usage
    bench_fn = lambda: process_samples(dataset, model, tokenizer)
    n_repeats = 2 if cfg["DEBUG"] else cfg["N_INFERENCE_REPEATS"]
    outputs, times, memories = do_bench(
        bench_fn=bench_fn,
        model_path=model_path,
        n_repeats=n_repeats,
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
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM benchmarks.")
    parser.add_argument(
        "--remote_env_path",
        "-r",
        type=str,
        required=True,
        help="Path to the remote environment file storing the encryption key.",
    )
    args = parser.parse_args()
    main(args)

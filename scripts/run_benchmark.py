import os
import gc
import argparse    
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
# from llama_cpp import Llama
# from vllm import LLM
from datasets import Dataset
from functools import partial
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
    # Force a fresh run for each new process
    torch_mp.set_start_method("spawn", force=True)
    
    # Load configurations from YAML files
    run_cfg = load_config_files(args)

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
        input_label_map={"Anonymised letter": "text", "Label_student": "label"}
    )

    # Benchmark the chosen model
    dataset = Dataset.from_pandas(df_data)
    if args.debug: dataset = Dataset.from_dict(mapping=dataset[:5])
    run_kwargs = {"cfg": run_cfg, "dataset": dataset, "debug": args.debug}

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
    print(f"Benchmarking {cfg['MODEL_PATH']} with {cfg['INFERENCE_BACKEND']} backend")

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
        raise(e)  # this will return None

    if model is None:
        print("Model is None, exiting benchmark.")
        return

    # Build prompts using the model tokenizer
    # dataset = dataset.copy()  # to avoid modifying the original dataset
    dataset = dataset.map(
        function=partial(func=build_prompt, cfg=cfg, tokenizer=tokenizer),
        desc="Building prompts",
    )

    # Record results obtained with the selected model
    n_inference_repeats = 2 if debug else cfg["N_INFERENCE_REPEATS"]
    benchmark_results = benchmark_one_model(
        dataset=dataset,
        model=model,
        model_path=cfg["MODEL_PATH"],
        tokenizer=tokenizer,
        n_inference_repeats=n_inference_repeats,
    )

    # Compute and save benchmark results
    print("Benchmarked %s" % cfg["MODEL_PATH"])
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
    output_subdir = cfg["INFERENCE_BACKEND"]
    if cfg["USE_OUTPUT_GUIDE"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["RESULT_DIR"], output_subdir)
    model_result_path = f"{cfg['MODEL_PATH']}-{cfg['QUANT_SCHEME']}.csv"
    output_path = os.path.join(output_dir, model_result_path)
    
    # Save model results to a csv file
    compute_and_save_metrics(
        benchmark_results=benchmark_results,
        model_path=cfg["MODEL_PATH"],
        output_path=output_path,
    )
    
    # Metrics are plotted by loading saved benchmark results
    metric_path = output_path.replace(".csv", ".json")
    plot_metrics(metric_path=metric_path)


def benchmark_one_model(
    dataset: Dataset,
    model: AutoModelForCausalLM,  # |LLM|Llama,
    model_path: str,
    tokenizer: AutoTokenizer,
    n_inference_repeats: int,
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
    # Measure computation time and GPU memory usage
    bench_fn = lambda: process_samples(dataset, model, tokenizer)
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
        description="Read an encrypted pandas DataFrame by fetching the key from a remote server."
    )

    add_model_arguments(parser)
    add_data_arguments(parser)
    
    args = parser.parse_args()
    main(args)


# python -m src.data.encryption \
#     -H "10.195.108.106" \
#     -u borneta
#     -r "/home/borneta/Documents/gemini/.env"
#     -f ./data/data_2025/processed/dataset.encrypted.csv
#     -k GEMINI
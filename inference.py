import os
import re
import json
import itertools
import gc
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from vllm import LLM, SamplingParams
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from utils import (
    do_bench, compute_and_save_metrics, plot_metrics,
    print_gpu_info, get_tokenizer_name, download_gguf_by_quant,
)
from config import Config as cfg, parse_script_args, build_prompts, get_output_guide

script_args = parse_script_args()
RUNTYPE = script_args.runtype.lower()
DEBUG = script_args.debug
PLOT_ONLY = script_args.plot_only


def main():
    """ Run benchmarks on different generative large language models in separate
        processes to avoid GPU memory leak or accumulation
    """
    # Select runs based on script argument
    if RUNTYPE == "all":
        run_args_list = list(itertools.chain(*cfg.RUN_DICT.values()))
    else:
        assert RUNTYPE in cfg.RUN_DICT, "Invalid runtype argument"
        run_args_list = cfg.RUN_DICT[RUNTYPE]
    
    # Benchmark all models sequentially, spawning one process per benchmark
    print(f"Processing {RUNTYPE} models\n")
    for run_args in run_args_list:
        if DEBUG or PLOT_ONLY:
            record_one_benchmark(run_args)
        else:
            process = torch_mp.Process(target=record_one_benchmark, args=(run_args,))
            process.start()  # spawn a new process for each benchmark run
            process.join()   # wait for the process to complete before continuing

    # Success message
    print("\nScript completed successfully!")


def record_one_benchmark(run_args: dict[str, str]) -> None:
    """ Runs the benchmark for a single model and record metrics
    """
    # Build unique output path given configuration
    model_path = run_args["model_path"]
    quant_scheme = run_args.get("quant_scheme", "no_quant_scheme")
    output_subdir = cfg.INFERENCE_BACKEND
    if cfg.VLLM_USE_OUTPUT_GUIDE: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg.RESULT_DIR, output_subdir)
    output_path = os.path.join(output_dir, f"{model_path}-{quant_scheme}.csv")
    
    # Do not the model if it's just for replotting the data
    if PLOT_ONLY:
        print("Replotting data for %s" % model_path)
    
    # Actual benchmark run
    else:
        print_gpu_info()
        print(f"Benchmarking {model_path} with {cfg.INFERENCE_BACKEND} backend")
        
        # Initialize model with the required backend
        try:
            model, tokenizer = get_model_and_tokenizer(**run_args)
        except ValueError as e:
            print(f"The following exception occurred: {e}. Skipping to next model.")
            return None
        
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
    if "model" in locals(): del model
    if torch_dist.is_initialized(): torch_dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_one_model(
    model: AutoModelForCausalLM|LLM|Llama,
    model_path: str,
    tokenizer: AutoTokenizer,
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
    # Pre-process dataset
    dataset = Dataset.from_csv(path_or_paths=cfg.DATASET_PATH)
    if DEBUG: dataset = Dataset.from_dict(mapping=dataset[:5])
    dataset = dataset.map(partial(build_prompts, tokenizer=tokenizer), desc="Building prompts")

    # Measure computation time and GPU memory usage
    bench_fn = lambda: process_samples(dataset, model, tokenizer)
    n_repeats = 2 if DEBUG else cfg.N_INFERENCE_REPEATS
    outputs, times, memories = do_bench(
        bench_fn=bench_fn,
        model_path=model_path,
        n_repeats=n_repeats,
        return_outputs=True,
    )
    times = times / len(dataset)  # since we want time per sample
    
    # Combine all outputs and add them to the input dataset
    for key in ["reasoning", "prediction"]:
        for i, output in enumerate(outputs):
            dataset = dataset.add_column(f"{key}_{i:03}", output[key])
    
    # Return benchmark results for metric computation and plotting
    return {"dataset": dataset, "times": times, "memories": memories}
    

def get_model_and_tokenizer(
    model_path: str,
    quant: str,
    quant_scheme: str|None=None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """ Create an LLM-based inference generator for solving a task
    
    Args:
        model_path (str): reference string to load model from huggingface
        quant (str): model file format (normal, bitsandbytes, awq, tqdm, gguf, etc.)
    """
    tokenizer = None
    match cfg.INFERENCE_BACKEND:
        
        # Using vLLM backend
        case "vllm":
            model_args = {
                "trust_remote_code": True,
                "max_model_len": cfg.MAX_CONTEXT_LENGTH,
            }
            if quant == "bnb":
                raise ValueError(f"vLLM does not support format {quant}")
            elif quant == "gguf":
                model_file_path = download_gguf_by_quant(model_path, quant_scheme)
                tokenizer_path = get_tokenizer_name(model_path)
                model_args.update({"model": model_file_path, "tokenizer": tokenizer_path})
            else:
                model_args.update({"model": model_path, "quantization": quant})
            model = LLM(**model_args)
        
        # Using Llama-cpp backend
        case "llama-cpp":
            if quant != "gguf":
                raise ValueError(f"Llama-cpp does not support format {quant}")
            model = Llama.from_pretrained(
                repo_id=model_path,
                filename=f"*{quant_scheme}*.gguf",
                n_gpu_layers=-1,
                n_ctx=cfg.MAX_CONTEXT_LENGTH,
                flash_attn=cfg.USE_FLASH_ATTENTION,
                verbose=False,
            )

        # Using HuggingFace backend
        case "hf":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                attn_implementation="flash_attention_2" if cfg.USE_FLASH_ATTENTION else None,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM|Llama|LLM,
    tokenizer: AutoTokenizer,
) -> dict[str, str]:
    """ Process a sample by formatting the input text, prompting an LLM, and
        extracting reasoning and predictions.

    Args:
        sample (dict[str, str]): sample including input text
        model (AutoModelForCausalLM): language model used for inference
        tokenizer (AutoTokenizer): tokenizer used by the inference model

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    # Use LLM inference to process the dataset
    output_texts = []
    match cfg.INFERENCE_BACKEND:

        case "vllm":
            sampling_params = SamplingParams(
                max_tokens=cfg.MAX_GENERATED_TOKENS,
                temperature=cfg.TEMPERATURE,
                top_p=cfg.TOP_P,
                guided_decoding=get_output_guide() if cfg.VLLM_USE_OUTPUT_GUIDE else None,
            )
            outputs = model.chat(dataset["messages"], sampling_params=sampling_params)
            output_texts = [output.outputs[0].text.strip() for output in outputs]
            
        case "llama-cpp":
            for messages in tqdm(dataset["messages"], desc="Generating inferences"):
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=cfg.MAX_GENERATED_TOKENS,
                    temperature=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                )
                output_text = response["choices"][0]["message"]["content"].strip()
                output_texts.append(output_text)
                
        case "huggingface":
            for prompt in tqdm(dataset["prompt"], desc="Generating inferences"):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=cfg.MAX_GENERATED_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    temp=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                )
                generated_tokens = output[0, inputs["input_ids"].shape[-1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                output_texts.append(output_text)
    
    # Update the dataset with the output the LLM
    dataset = dataset.add_column("output_text", output_texts)
    dataset = dataset.map(
        lambda s: extract_reasoning_and_prediction(s["output_text"]),
        desc="Extracting model predictions",
    )

    return dataset


def extract_reasoning_and_prediction(raw_output: str) -> dict:
    """ Extract reasoning and mRS score from raw model output
    
        Args:
            raw_output_text (str): raw output from the LLM

        Returns:
            dict[str, str]: structured output from the LLM
    """
    # Try direct JSON parse
    try:
        formatted_output = json.loads(raw_output.strip())
        return normalize_output_keys(formatted_output)
    except json.JSONDecodeError:
        print("Parsing LLM output failed, falling back to more lenient parsing")
        pass

    # Try extracting substring between first "{" and last "}"
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        json_candidate = raw_output[start:end]
        formatted_output = json.loads(json_candidate)
        return normalize_output_keys(formatted_output)
    except (ValueError, json.JSONDecodeError):
        print("Lenient parsing failed as well, falling back to regex matching")
        pass  # continue to fallback 2

    # Fallback on regex-based strategies
    matches = list(re.finditer(r"mRS[\s:;-]{0,10}([0-6])\b", raw_output, re.IGNORECASE))
    if matches:
        mrs_score = int(matches[-1].group(1))  # last matching digit close to "mRS"
    else:
        all_scores = re.findall(r"\b[0-6]\b", raw_output)  # ast digit number
        mrs_score = int(all_scores[-1]) if all_scores else -1
    
    formatted_output = {"reasoning": raw_output.strip(), "prediction": mrs_score}
    return normalize_output_keys(formatted_output)


def normalize_output_keys(output_dict: dict) -> dict:
    """ Normalize formatted LLM output for mRS extraction
    """
    reasoning = ""  # indicating "no reasoning"
    prediction = -1  # indicating "mRS not found"
    for key, value in output_dict.items():
        if key.lower() == "reasoning":
            reasoning = value
        if key.lower() == "mrs":
            if -1 <= value <= 6:
                prediction = value

    return {"reasoning": reasoning, "prediction": prediction}
    

if __name__ == "__main__":
    if not DEBUG and not PLOT_ONLY: torch_mp.set_start_method("spawn", force=True)
    main()
    
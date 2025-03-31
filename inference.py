import os
import re
import gc
import itertools
import numpy as np
import torch
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from vllm import LLM, SamplingParams
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import do_bench, record_metrics, print_gpu_info, get_tokenizer_name, download_gguf_by_quant
from config import Config as cfg, parse_script_args, build_prompts

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
            clean_benchmark_run(run_args)
        else:
            process = torch_mp.Process(target=clean_benchmark_run, args=(run_args,))
            process.start()  # spawn a new process for each benchmark run
            process.join()   # wait for the process to complete before continuing

    # Success message
    print("\nScript completed successfully!")


def clean_benchmark_run(run_args: dict[str, str]) -> None:
    """ Runs the benchmark for a single model in a separate process
    """
    # Run benchmark
    try:
        model_path = run_args["model_path"]
        output_dir = os.path.join(cfg.RESULT_DIR, cfg.INFERENCE_BACKEND)
        output_path = os.path.join(output_dir, f"{model_path}_raw.csv")
        if PLOT_ONLY:
            print("Replotting data for %s" % model_path)
            metrics = None  # flag metrics to be loaded from pickle file
        else:
            print_gpu_info()
            print(f"Benchmarking {model_path} with {cfg.INFERENCE_BACKEND} backend")
            model, tokenizer = get_model_and_tokenizer(**run_args)
            metrics = benchmark_one_model(
                model=model,
                tokenizer=tokenizer,
                output_path=output_path,
            )
            print("Benchmarked %s" % model_path)
        record_metrics(output_path, metrics=metrics)
    
    # Skip invalid models
    except AssertionError as e:
        print("Model not eligible for current backend. Skipping to next one.")
        
    # Data cleaning part to free GPU memory
    finally:
        if "model" in locals(): del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        
def benchmark_one_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str,
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
    times, memories, outputs = do_bench(bench_fn=bench_fn, n_repeats=n_repeats, return_outputs=True)
    times = times / len(dataset)  # since we want time per sample
    
    # Compute performance metrics
    cm_fn = lambda d: confusion_matrix(d["label"], d["prediction"], labels=list(range(-1, 7)))
    error_fn = lambda d: np.mean(np.array(d["prediction"]) != np.array(d["label"]))
    distance_fn = lambda d: np.mean(np.abs(np.array(d["prediction"]) - np.array(d["label"])))
    confusion_matrix_results = np.sum([cm_fn(o) for o in outputs], axis=0)
    errors = torch.tensor([error_fn(o) for o in outputs], dtype=torch.float32)
    distances = torch.tensor([distance_fn(o) for o in outputs], dtype=torch.float32)
    
    # Combine all outputs and add them to the input dataset
    for key in ["reasoning", "answer", "prediction"]:
        for i, output in enumerate(outputs):
            dataset = dataset.add_column(f"{key}_{i:03}", output[key])
            
    # Write raw results a csv file
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    # Return computed metrics for plotting
    return {
        "Time per Sample": {"unit": "s", "max_y": 100.0, "values": times},
        "Peak VRAM Usage": {"unit": "GB", "max_y": 60.0, "values": memories},
        "Error Rate": {"unit": "%", "max_y": 1.0, "values": errors},
        "Distance": {"unit": "mRS", "max_y": 2.0, "values": distances},
        "Confusion Matrix": confusion_matrix_results,
    }


def get_model_and_tokenizer(
    model_path: str,
    quant: str,
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
                "gpu_memory_utilization": 0.95,
            }
            if quant == "bnb":
                raise ValueError(f"vLLM does not support format {quant}")
            elif quant == "gguf":
                model_file_path = download_gguf_by_quant(model_path, cfg.GGUF_QUANT_SCHEME)
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
                filename=f"*{cfg.GGUF_QUANT_SCHEME}*.gguf",
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


def extract_reasoning_and_prediction(
    llm_raw_output_text: str,
) -> dict[str, str]:
    """ Extract reasoning and answer from the raw output of an LLM
    
    Args:
        raw_output_text (str): raw output from the LLM

    Returns:
        dict[str, str]: structured output from the LLM
    """
    # Extract reasoning and text answer from raw output 
    lines = llm_raw_output_text.split("\n")    
    reasoning = "\n".join(lines[:-1])
    answer = lines[-1].strip()
    
    # Extract prediction from text answer
    pattern = re.compile(r'(?:mrs[:\s]*|\b)([0-6])\b', re.IGNORECASE)
    match = pattern.search(answer)
    prediction = int(match.group(1)) if match else -1  # -1 being "bad answer"
    
    return {"reasoning": reasoning, "answer": answer, "prediction": prediction}


if __name__ == "__main__":
    if not DEBUG: torch_mp.set_start_method("spawn", force=True)
    main()
    
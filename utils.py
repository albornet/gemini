import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from tqdm import tqdm
from warnings import warn
from typing import Any, Union, Callable
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files, hf_hub_download, HfApi


def do_bench(
    bench_fn: Callable,
    n_repeats: int=10,
    return_outputs: bool=False,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]|None]:
    """ Benchmarking function modified from triton.testing to return output
        https://triton-lang.org/main/python-api/generated/triton.testing.do_bench 
    
    Params:
        bench_fn: function to benchmark
        n_repeats: number of repetitions for benchmark
        return_outputs: if True, return outputs of the benchmarked function
    """
    assert n_repeats > 1, "n_repeats must be greater than 1"
    
    # Initialize events, results storage, and outputs (if required)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = torch.empty(n_repeats, dtype=torch.float)
    memories = torch.empty(n_repeats, dtype=torch.float)
    outputs = [] if return_outputs else None
    
    # Benchmark loop
    for i in tqdm(range(n_repeats), desc="Benchmarking inference model"):

        # Clear L2 cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Time the function execution
        start_event.record()
        output = bench_fn()
        end_event.record()
        torch.cuda.synchronize()
        
        # Record the time and memory usage
        times[i] = start_event.elapsed_time(end_event) / 1000  # in seconds
        memories[i] = get_gpu_memory_usage_by_pid()  # in GB

        # Collect output if requested
        if return_outputs:
            outputs.append(output)
    
    return times, memories, outputs


def get_gpu_memory_usage_by_pid():
    """ Compute the amount of GPU memory used at the moment by the current PID
    
    Returns:
        float: GPU memory used in all available GPUs, in GB
    """
    # Collect the current process ID and the output of nvidia-smi
    pid = str(os.getpid())
    cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,nounits,noheader"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    
    # Extract the amount of GPU memory used by the current PID
    MiB_used_by_pid = 0
    for line in result.stdout.strip().split('\n'):
        if not line.strip(): continue
        proc_pid, mem = [x.strip() for x in line.split(',')]
        if proc_pid == pid: MiB_used_by_pid += int(mem)
    
    # Convert to GB and return
    GB_used_by_pid = MiB_used_by_pid * 1024 ** 2 / 1000 ** 3
    return GB_used_by_pid


def record_metrics(
    output_path: str,  # without extension
    metrics: dict[str, Union[torch.Tensor, np.ndarray]]|None,
) -> None:
    """ Record metrics (or load them) and plot them to a nice figure
    """
    # Record metrics as a pickle file or load them from a previous run
    result_path = output_path.replace("_raw.csv", "_metrics.pkl")
    if metrics is not None:
        with open(result_path, "wb") as f: pickle.dump(metrics, f)
    else:
        with open(result_path, "rb") as f: metrics = pickle.load(f)
    confusion_matrix = metrics.pop("Confusion Matrix")

    # Determine the number of subplots
    num_metrics = len(metrics)
    _, ax = plt.subplots(
        nrows=1,
        ncols=num_metrics + 1,  # "+ 1" for the confusion matrix
        figsize=(12, 5),
        width_ratios=[0.5] + [0.5 / num_metrics for _ in range(num_metrics)],
    )

    # Plot confusion matrix
    labels = range(-1, len(confusion_matrix) - 1)  # [-1, 0, 1, 2, 3, 4, 5, 6]
    ax[0].imshow(confusion_matrix, cmap="Blues", interpolation="none")
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_yticks(range(len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Predicted mRS")
    ax[0].set_ylabel("True mRS")
    ax[0].set_title("Confusion Matrix")
    
    # Polish confusion matrix
    max_cm_value = np.max(confusion_matrix)
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            value = confusion_matrix[i, j]
            if value > 0:
                color = "white" if value > max_cm_value / 2 else "black"
                ax[0].text(j, i, str(value), ha="center", va="center", color=color)
            
    # Plot each metric in a separate subplot
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"] * 10
    for i, (metric_name, metric_dict) in enumerate(metrics.items()):
        mean = metric_dict["values"].mean().item()
        sem = metric_dict["values"].std(unbiased=True).item() / len(metric_dict["values"])
        ax[i + 1].bar(0, mean, yerr=sem, capsize=5, alpha=0.75, color=colors[i])
        ax[i + 1].set_xticks([])
        ax[i + 1].set_ylim([0.0, metric_dict["max_y"]])
        ax[i + 1].set_ylabel("[%s]" % metric_dict["unit"])
        ax[i + 1].set_title(metric_name)
    
    # Adjust layout and save plot
    plot_path = result_path.replace(".pkl", ".png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def print_gpu_info():
    """ Print information about available GPU(s)
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("GPU Information:\n")
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.")


def get_tokenizer_name(
    model_id: str,
    chat_template_required: bool=True,
) -> str:
    """ Identify base model from which any model was quantized, in order to load
        the correct tokenizer
    """
    # Look for base model in the "cardData" (where model tree info is stored)
    api = HfApi()
    info = api.model_info(model_id)
    card_data = info.card_data or {}
    tokenizer_name = card_data.get("base_model")
    
    # Alternatively, inspect tags or siblings
    if not tokenizer_name:
        for tag in info.tags:
            if "base_model:" in tag:
                tokenizer_name = tag.split(":")[1]
                break
    
    # Check for chat template, may fall back on Llama-3.2-3B-Instruct (most common)
    if chat_template_required and tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.chat_template is None:
                raise ValueError("chat_template missing")
        except Exception as e:
            warn(
                f"The tokenizer '{tokenizer_name}' does not support chat templating "
                f"(reason: {e}). Falling back to 'meta-llama/Llama-3.2-3B-Instruct'."
            )
            tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"

    return tokenizer_name


def download_gguf_by_quant(model_id: str, quant: str) -> str:
    """ Download the first matching GGUF file in a model repository
    """
    files = list_repo_files(model_id)
    for file in files:
        if file.endswith(".gguf") and quant in file:
            return hf_hub_download(repo_id=model_id, filename=file)
        
    raise FileNotFoundError(f"No GGUF file found with quantization scheme '{quant}' in repo '{model_id}'")

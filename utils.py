import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from typing import Any, Callable


def do_bench_custom(
    benchmarked_fn: Callable,
    n_warmups: int=0,
    n_repeats: int=10,
    return_outputs: bool=False,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]|None]:
    """ Benchmarking function modified from triton.testing to return output
        https://triton-lang.org/main/python-api/generated/triton.testing.do_bench 
        
    :param benchmarked_fn: function to benchmark
    :param n_warmups: number of warmup steps
    :param n_repeats: number of repetitions for benchmark
    :param return_outputs: if True, return outputs of the benchmarked function
    :param grad_to_none: reset the gradient of the provided tensor to None
    """
    assert n_repeats > 1, "Quantiles cannot be measured for n_repeats < 1"
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    
    # Compute number of warmup and repeat
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
    
    # Warm-up
    for _ in range(n_warmups):
        benchmarked_fn()
        
    # Benchmark
    outputs = [] if return_outputs else None
    times = torch.empty(n_repeats, dtype=torch.float)
    memories = torch.empty(n_repeats, dtype=torch.float)
    for i in range(n_repeats):
        
        # Clear L2 cache and reset peak memory usage counter
        cache.zero_()
        torch.cuda.reset_peak_memory_stats()
                
        # Record time of fn
        start_event[i].record()
        output = benchmarked_fn()
        end_event[i].record()
        
        # Record time and peak memory usage
        torch.cuda.synchronize()
        times[i] = start_event[i].elapsed_time(end_event[i]) / 1000  # in seconds
        max_gpu_memory = sum([
            torch.cuda.max_memory_allocated(device=d) / (1024 ** 3)  # in GB
            for d in range(torch.cuda.device_count())
        ])
        if max_gpu_memory < 0.5:
            max_gpu_memory = get_memory_usage_without_torch()
        memories[i] = max_gpu_memory
        
        # Capture metric using fn output
        if return_outputs:
            outputs.append(output)
    
    return times, memories, outputs


def get_memory_usage_without_torch():
    """ Compute the amount of GPU memory used at the moment
    
    Returns:
        int: GPU memory used in all available GPUs, in GB
    """
    # Run `nvidia-smi` and get the output
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True
    )
    
    # Convert the result to an integer, in GB)
    MiB_per_device = [int(s.strip()) for s in result.stdout.split("\n") if s]
    MiB_used = sum(MiB_per_device)
    GB_used = MiB_used * 1024 ** 2 / 1000 ** 3
    
    return GB_used


def record_metrics(
    output_path: str,  # without extension
    confusion_matrix: np.ndarray,
    metrics: dict[str, torch.Tensor],
) -> None:
    """ Record metrics and confusion matrix as csv and png files
    """
    # Record metrics in a csv file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path + ".csv", index=False)
    
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
    
    # Plot each metric in a separate subplot
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"] * 10
    for i, metric in enumerate(metrics):
        mean = metric["values"].mean().item()
        sem = metric["values"].std(unbiased=True).item() / len(metric["values"])
        ax[i + 1].bar(0, mean, yerr=sem, capsize=5, alpha=0.75, color=colors[i])
        ax[i + 1].set_xticks([])
        ax[i + 1].set_ylim([0.0, metric["max_y"]])
        ax[i + 1].set_ylabel("[%s]" % metric["unit"])
        ax[i + 1].set_title(metric["name"])
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(output_path + ".png", dpi=300)
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
        
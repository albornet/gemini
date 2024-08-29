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
    grad_to_none: torch.Tensor=None,
    fast_flush: bool=True,
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
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn"t contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    
    # Compute number of warmup and repeat
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeats)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeats)]
    
    # Warm-up
    for _ in range(n_warmups):
        benchmarked_fn()
        
    # Benchmark
    outputs = [] if return_outputs else None
    times = torch.empty(n_repeats, dtype=torch.float)
    memories = torch.empty(n_repeats, dtype=torch.float)
    for i in range(n_repeats):
        # We don"t want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
                
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
        memories[i] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
        
        # Capture metric using fn output
        if return_outputs:
            outputs.append(output)
    
    return times, memories, outputs


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
        ncols=num_metrics + 1,  # +1 for the confusion matrix
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
    ax[0].set_xlabel("True mRS")
    ax[0].set_ylabel("Predicted mRS")
    ax[0].set_title("Confusion Matrix")
    
    # Plot each metric in a separate subplot
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"] * 10
    # for i, (metric_name, mean, sem) in enumerate(zip(names, max_ys, means, sems)):
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
        
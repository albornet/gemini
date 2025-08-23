import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from tqdm import tqdm
from typing import Any, Callable
from collections import Counter
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import confusion_matrix


def do_bench(
    bench_fn: Callable,
    model_path: str,
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
    # Initialize events, results storage, and outputs (if required)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = torch.empty(n_repeats, dtype=torch.float)
    memories = torch.empty(n_repeats, dtype=torch.float)
    outputs = [] if return_outputs else None
    
    # Benchmark loop
    for i in tqdm(range(n_repeats), desc=f"Benchmarking {model_path}"):

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
    
    return outputs, times, memories


def compute_and_save_metrics(
    benchmark_results: dict,
    model_path: str,
    output_path: str,
) -> dict:
    """ Compute metrics for one set of model predictions and labels
    """
    # Write inputs and corresponding raw model outputs to a csv file
    dataset: Dataset = benchmark_results.pop("dataset")
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved raw results at {output_path}")

    # Pool model predictions and plot confusion matrices for both pooling strategies
    preds_and_labels = extract_preds_and_labels(dataset)
    y_true_all, y_pred_all = pool_model_predictions(preds_and_labels, "concatenation")
    y_true_single, y_pred_single = pool_model_predictions(preds_and_labels, "single")
    y_true_maj_3, y_pred_maj_3 = pool_model_predictions(preds_and_labels, "single", num_models=3)
    y_true_maj_5, y_pred_maj_5 = pool_model_predictions(preds_and_labels, "majority", num_models=5)
    y_true_maj_10, y_pred_maj_10 = pool_model_predictions(preds_and_labels, "majority", num_models=10)

    # Basic model characteristics
    num_params = get_model_number_of_parameters(model_path)
    bits_per_param = get_model_bits_per_parameter(model_path, output_path)
    total_bits = num_params * bits_per_param
    params_weight = total_bits / (8 * 1024**3)

    # Record performance and efficiency metrics
    metric_dict = {

        # True labels for different voting strategies
        "y_true": {
            "all": y_true_all, "single": y_true_single,
            "maj_3": y_true_maj_3, "maj_5": y_true_maj_5, "maj_10": y_true_maj_10,
        },

        # Predicted labels for different voting strategies
        "y_pred": {
            "all": y_pred_all, "single": y_pred_single,
            "maj_3": y_pred_maj_3, "maj_5": y_pred_maj_5, "maj_10": y_pred_maj_10,
        },

        # How often the model had it wrong, on average
        f"Error Rate\n(all {len(preds_and_labels)} models)": {
            "values": torch.tensor([np.mean(np.array(o["mRS"]) != o["label"]) for o in preds_and_labels]),
            "unit": "%", "max_y": 1.0, "loc": (1, 1), "color": "tab:red",
        },
        "Error Rate\n(single model)": {
            "values": np.mean(y_true_single != y_pred_single, keepdims=True),
            "unit": "%", "max_y": 1.0, "loc": (2, 1), "color": "tab:red",
        },
        "Error Rate\n(maj-pooling-3)": {
            "values": np.mean(y_true_maj_3 != y_pred_maj_3, keepdims=True),
            "unit": "%", "max_y": 1.0, "loc": (3, 1), "color": "tab:red",
        },
        "Error Rate\n(maj-pooling-5)": {
            "values": np.mean(y_true_maj_5 != y_pred_maj_5, keepdims=True),
            "unit": "%", "max_y": 1.0, "loc": (4, 1), "color": "tab:red",
        },
        "Error Rate\n(maj-pooling-10)": {
            "values": np.mean(y_true_maj_10 != y_pred_maj_10, keepdims=True),
            "unit": "%", "max_y": 1.0, "loc": (5, 1), "color": "tab:red",
        },

        # How far from the correct label the model was, on average
        f"Distance\n(all {len(preds_and_labels)} models)": {
            "values": [np.mean(np.abs(np.array(o["mRS"]) - o["label"])) for o in preds_and_labels],
            "unit": "mRS", "max_y": 10.0, "loc": (1, 2), "color": "tab:orange",
        },
        "Distance\n(single model)": {
            "values": np.mean(np.abs(y_true_single - y_pred_single), keepdims=True),
            "unit": "mRS", "max_y": 10.0, "loc": (2, 2), "color": "tab:orange",
        },
        "Distance\n(maj-pooling-3)": {
            "values": np.mean(np.abs(y_true_maj_3 - y_pred_maj_3), keepdims=True),
            "unit": "mRS", "max_y": 10.0, "loc": (3, 2), "color": "tab:orange",
        },
        "Distance\n(maj-pooling-5)": {
            "values": np.mean(np.abs(y_true_maj_5 - y_pred_maj_5), keepdims=True),
            "unit": "mRS", "max_y": 10.0, "loc": (4, 2), "color": "tab:orange",
        },
        "Distance\n(maj-pooling-10)": {
            "values": np.mean(np.abs(y_true_maj_10 - y_pred_maj_10), keepdims=True),
            "unit": "mRS", "max_y": 10.0, "loc": (5, 2), "color": "tab:orange",
        },
        
        # Infrastructure metrics
        "Number of params": {
            "values": np.array([num_params / 10**9]),
            "unit": "Billions", "max_y": 75, "loc": (0, 0), "color": "tab:gray",
        },
        "Bits/param": {
            "values": np.array([bits_per_param]),
            "unit": "Bits", "max_y": 16, "loc": (0, 1), "color": "tab:brown",
        },
        "VRAM param usage": {
            # "values": benchmark_results["memories"],
            "values": np.array([params_weight]),
            "unit": "GB", "max_y": 60.0, "loc": (0, 2), "color": "tab:blue",
        },
        "Time/sample": {
            "values": benchmark_results["times"],
            "unit": "s", "max_y": 100.0, "loc": (0, 3), "color": "tab:green",
        },

    }

    # Save plotted data to a handy json file (for later pooled figures)
    metric_dict = convert_to_json_serializable(metric_dict)
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
        print(f"Saved metrics summary at {json_path}")

    return metric_dict


def extract_preds_and_labels(dataset: Dataset):
    """ Extract a set of model predictions and output labels for different runs
        to a list of datasets with labels and predictions
    """
    num_models = sum(["mRS_" in f for f in dataset.features])
    preds_and_labels = [{"label": dataset["label"]} for _ in range(num_models)]
    for feature in dataset.features:
        if "mRS_" in feature:
            original_feature_name, index = feature.split("_")
            preds_and_labels[int(index)][original_feature_name] = dataset[feature]

    return [Dataset.from_dict(dataset) for dataset in preds_and_labels]


def convert_to_json_serializable(obj):
    """ Recursively converts arrays and tensors in a dictionary to lists
    """
    # Dict case is made to go deeper
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    
    # Types to convert to numbers (arrays, tensors)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, tuple): # JSON doesn't have tuples, convert to list
        return list(obj)
    
    return obj


def pool_model_predictions(
    preds_and_labels: list[Dataset],
    pred_pool_mode: str,
    num_models: int|None=None,
) -> tuple[np.ndarray]:
    """ Pool the predictions of several models on the same set of samples, given
        the pooling method
    """
    # Select only a fraction of the models, if required
    if num_models is not None and num_models > 0:
        preds_and_labels = preds_and_labels[:num_models]
    
    # Single model prediction (taking only the first one)
    if pred_pool_mode == "single":
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = preds_and_labels[0]["mRS"]

    # Pool model predictions by concatenating them (hence, sum in the confusion matrix)
    elif pred_pool_mode == "concatenation":
        y_true_pooled = sum([o["label"] for o in preds_and_labels], [])
        y_pred_pooled = sum([o["mRS"] for o in preds_and_labels], [])

    # Pool model predictions by taking the vote of the majority
    elif pred_pool_mode == "majority":
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = []
        num_samples = preds_and_labels[0].num_rows
        preds_by_model = [dataset["mRS"] for dataset in preds_and_labels]
        for i in range(num_samples):
            votes = [preds[i] for preds in preds_by_model]
            y_pred_pooled.append(Counter(votes).most_common(1)[0][0])
    
    # Unexpected pred_pool_mode value
    else:
        raise ValueError("Invalid pooling mode (single, concatenation, majority)")

    return np.array(y_true_pooled), np.array(y_pred_pooled)


def plot_metrics(metric_path: str) -> None:
    """ Plot metrics in a common plot for different prediction voting strategies
    """
    # Subplots (one row with 4 small columns, 5 rows with 1 large and 2 small columns)
    fig = plt.figure(figsize=(7, 20))
    gs = fig.add_gridspec(
        nrows=6, ncols=4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1, 1, 1, 1, 1, 1],
    )
    ax = []
    for row_idx in range(6):
        if row_idx == 0:
            ax.append((
                fig.add_subplot(gs[row_idx, 0]), fig.add_subplot(gs[row_idx, 1]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))
        else:
            ax.append((
                fig.add_subplot(gs[row_idx, 0:2]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))
    
    # Load data to plot
    with open(metric_path, "r") as f:
        metric_dict: dict = json.load(f)
    y_true = metric_dict.pop("y_true")
    y_pred = metric_dict.pop("y_pred")
    
    # Plot all confusion matrices
    plot_cm(ax=ax[1][0], y_true=y_true["all"], y_pred=y_pred["all"], title_flag=f"all models")
    plot_cm(ax=ax[2][0], y_true=y_true["single"], y_pred=y_pred["single"], title_flag="single model")
    plot_cm(ax=ax[3][0], y_true=y_true["maj_3"], y_pred=y_pred["maj_3"], title_flag="maj-pooling-3")
    plot_cm(ax=ax[4][0], y_true=y_true["maj_5"], y_pred=y_pred["maj_5"], title_flag="maj-pooling-5")
    plot_cm(ax=ax[5][0], y_true=y_true["maj_10"], y_pred=y_pred["maj_10"], title_flag="maj-pooling-10")

    # Plot each metric in a separate subplot
    for metric_name, plot_dict in metric_dict.items():

        # Compute mean values
        plot_values = plot_dict["values"]
        mean = np.mean(plot_values)

        # Compute standard error of the mean for error bar (if enough samples)
        bar_kwargs = {"capsize": 5, "alpha": 0.75, "color": plot_dict["color"]}
        if len(plot_values) > 1:
            bar_kwargs["yerr"] = np.std(plot_values, ddof=1) / np.sqrt(len(plot_values))
        else:
            bar_kwargs["yerr"] = None
        
        # Plot the bar, with or without error bar
        i, j = plot_dict["loc"]
        ax[i][j].bar(0, mean, **bar_kwargs)
        ax[i][j].set_xticks([])
        ax[i][j].set_ylim([0.0, plot_dict["max_y"]])
        ax[i][j].set_ylabel(f"[{plot_dict['unit']}]")
        ax[i][j].set_title(metric_name.split("\n")[0])

    # Adjust layout and save plot
    plot_path = metric_path.replace(".json", ".png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved processed result plot at {plot_path}")


def plot_cm(
    ax: plt.Axes,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title_flag: str,
    possible_labels: list[Any]=[-1, 0, 1, 2, 3, 4, 5, 6],  # mRS in this case
) -> None:
    """ Plot a confusion matrix given model prediction and true label values
    """
    # Plot pooled confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=possible_labels)
    ax.imshow(cm, cmap="Blues", interpolation="none")
    ax.set_xticks(range(len(possible_labels)))
    ax.set_yticks(range(len(possible_labels)))
    ax.set_xticklabels(possible_labels)
    ax.set_yticklabels(possible_labels)
    ax.set_xlabel("Predicted mRS")
    ax.set_ylabel("True mRS")
    ax.set_title(f"Confusion Matrix ({title_flag})")
    
    # Polish confusion matrix
    max_cm_value = np.max(cm)
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            value = cm[i, j]
            if value > 0:
                color = "white" if value > max_cm_value / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color)


def print_gpu_info():
    """ Print information about available GPU(s)
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("GPU Information:\n")
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.")


def get_model_number_of_parameters(model_id: str) -> int:
    """ Get the number of parameters in a huggingface model, using various methods
    """
    # Load model info from the huggingface API
    api = HfApi()
    model_info = api.model_info(model_id)

    # Try to identify the number of parameters assuming the model is a gguf file
    try:
        return model_info.gguf["total"]
    except Exception:
        print("Could not identify number of parameters using the gguf method")
        pass

    # Try to identify the number of parameters with a more classic method
    try:
        return model_info.safetensors.total
    except:
        print("Could not identify number of parameters using the hugginface method")
        return 0
    

def get_model_bits_per_parameter(
    model_path: str,
    output_path: str,
) -> int:
    """ Get the number of parameters using heuristics from my own file-naming system
    """
    # Identify quantization scheme
    scheme = output_path.split(model_path)[-1].strip("-").split(".")[0]
    scheme_check = output_path.split("-")[-1].split(".")[0]
    assert scheme == scheme_check

    # Try to identify the number of bits with my own heuristics
    if scheme == "no_quant_scheme": return 4  # for awq, etc.
    match = re.search(r"^(?:I?Q)(\d+)", scheme)
    if match: return int(match.group(1))
    if scheme.lower() in ["f16", "fp16"]: return 16
    if scheme.lower() == "q8_0": return 8
    return 0


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

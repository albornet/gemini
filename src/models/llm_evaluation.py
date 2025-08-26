import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from typing import Any
from collections import Counter
from collections.abc import Iterable
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import confusion_matrix

POOLED_MODES = [
    {"name": "all", "label": "all models", "pred_pool_mode": "concatenation"},
    {"name": "single", "label": "single model", "pred_pool_mode": "single"},
    {"name": "maj_3", "label": "maj-pooling-3", "pred_pool_mode": "majority", "num_models": 3},
    {"name": "maj_5", "label": "maj-pooling-5", "pred_pool_mode": "majority", "num_models": 5},
    {"name": "maj_10", "label": "maj-pooling-10", "pred_pool_mode": "majority", "num_models": 10},
]


def pool_model_predictions(
    preds_and_labels: list[Dataset],
    pred_pool_mode: str,
    num_models: int | None = None,
) -> tuple[np.ndarray]:
    """
    Pool model predictions on the same set of samples given a pooling method
    """
    if num_models is not None and 0 < num_models <= len(preds_and_labels):
        preds_and_labels = preds_and_labels[:num_models]

    if pred_pool_mode == "single":
        if not preds_and_labels: return np.array([]), np.array([])
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = preds_and_labels[0]["mRS"]

    elif pred_pool_mode == "concatenation":
        y_true_pooled = sum([[v for v in col["label"]] for col in preds_and_labels], [])
        y_pred_pooled = sum([[v for v in col["mRS"]] for col in preds_and_labels], [])

    elif pred_pool_mode == "majority":
        if not preds_and_labels: return np.array([]), np.array([])
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = []
        num_samples = preds_and_labels[0].num_rows
        preds_by_model = [dataset["mRS"] for dataset in preds_and_labels]

        for i in range(num_samples):
            votes = [preds[i] for preds in preds_by_model]
            y_pred_pooled.append(Counter(votes).most_common(1)[0][0])

    else:
        raise ValueError("Invalid pooling mode (single, concatenation, majority)")

    return np.array(y_true_pooled), np.array(y_pred_pooled)


def extract_preds_and_labels(dataset: Dataset) -> list[Dataset]:
    """
    Extract a set of model predictions and output labels for different runs
    to a list of datasets with labels and predictions
    """
    num_models = sum(1 for f in dataset.features if f.startswith("mRS_"))
    dataset = dataset.map(lambda x: {"label": -1 if x["label"] is None else int(x["label"])})
    preds_and_labels = [{"label": dataset["label"]} for _ in range(num_models)]

    for feature in dataset.features:
        if feature.startswith("mRS_"):
            original_feature_name, index = feature.split("_")
            preds_and_labels[int(index)][original_feature_name] = dataset[feature]

    return [Dataset.from_dict(d) for d in preds_and_labels]


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively converts arrays and tensors in a dictionary to lists.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    return obj


def _record_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    loc: tuple,
) -> dict:
    """
    Records performance metrics (Error Rate and Distance) for a given set of labels and predictions.
    """
    error_rate = np.mean(y_true != y_pred, keepdims=True)
    distance = np.mean(np.abs(y_true - y_pred), keepdims=True)

    return {
        f"Error Rate\n({label})": {
            "values": error_rate,
            "unit": "%", "max_y": 1.0, "loc": (loc[0], 1), "color": "tab:red",
        },
        f"Distance\n({label})": {
            "values": distance,
            "unit": "mRS", "max_y": 10.0, "loc": (loc[0], 2), "color": "tab:orange",
        },
    }


def _record_metrics_for_pooling_modes(
    preds_and_labels: list[Dataset],
    pooling_modes: list[dict],
) -> tuple[dict, dict]:
    """
    Computes metrics for different pooling modes, dynamically handling any number of models.
    """
    y_true_dict = {}
    y_pred_dict = {}
    metric_dict = {}

    for i, mode in enumerate(pooling_modes, start=1):
        mode_name = mode["name"]
        y_true, y_pred = pool_model_predictions(
            preds_and_labels,
            mode["pred_pool_mode"],
            mode.get("num_models"),
        )
        y_true_dict[mode_name] = y_true
        y_pred_dict[mode_name] = y_pred
        
        # Add a conditional check to avoid errors if y_true is empty
        if y_true.size > 0:
            metrics = _record_metrics(y_true, y_pred, mode["label"], loc=(i, 1))
            metric_dict.update(metrics)

    return y_true_dict, y_pred_dict, metric_dict



def compute_and_save_metrics(
    benchmark_results: dict,
    model_path: str,
    output_path: str,
) -> dict:
    """
    Compute and save metrics for a set of model predictions
    """
    # Save inputs and raw outputs to a CSV file
    dataset: Dataset = benchmark_results.pop("dataset")
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved raw results at {output_path}")

    # Extract predictions and labels
    preds_and_labels = extract_preds_and_labels(dataset)
    num_models = len(preds_and_labels)

    # Filter pooling modes based on the number of models
    pooling_modes = []
    for mode in POOLED_MODES:
        if mode.get("num_models", num_models) <= num_models:
            # For the 'all' mode, update the label with the actual number of models
            if mode["name"] == "all":
                mode = mode.copy() # Avoid modifying the global list
                mode["label"] = f"all {num_models} models"
            pooling_modes.append(mode)

    y_true_pooled, y_pred_pooled, pooled_metrics = _record_metrics_for_pooling_modes(
        preds_and_labels,
        pooling_modes,
    )

    # Basic model characteristics
    num_params = get_model_number_of_parameters(model_path)
    bits_per_param = get_model_bits_per_parameter(model_path, output_path)
    total_bits = num_params * bits_per_param
    params_weight = total_bits / (8 * 1024**3)

    # Build the final metric dictionary
    metric_dict = {
        "y_true": y_true_pooled,
        "y_pred": y_pred_pooled,
        **pooled_metrics,
        "Number of params": {
            "values": np.array([num_params / 10**9]),
            "unit": "Billions", "max_y": 75, "loc": (0, 0), "color": "tab:gray",
        },
        "Bits/param": {
            "values": np.array([bits_per_param]),
            "unit": "Bits", "max_y": 16, "loc": (0, 1), "color": "tab:brown",
        },
        "VRAM param usage": {
            "values": np.array([params_weight]),
            "unit": "GB", "max_y": 60.0, "loc": (0, 2), "color": "tab:blue",
        },
        "Time/sample": {
            "values": benchmark_results["time"],
            "unit": "s", "max_y": 100.0, "loc": (0, 3), "color": "tab:green",
        },
    }

    # Save plotted data to a handy json file
    metric_dict = convert_to_json_serializable(metric_dict)
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
        print(f"Saved metrics summary at {json_path}")

    return metric_dict


def plot_metrics(metric_path: str) -> None:
    """
    Plot metrics in a common plot for different prediction voting strategies
    """
    # Load data to plot to determine the number of rows needed
    with open(metric_path, "r") as f:
        metric_dict: dict = json.load(f)
    y_true = metric_dict.pop("y_true", {})
    y_pred = metric_dict.pop("y_pred", {})

    # Determine which rows will be plotted and build the confusion matrix config
    rows_to_plot = [0]
    cms_to_plot = []
    for mode_idx, mode in enumerate(POOLED_MODES):
        if mode["name"] in y_true and mode["name"] in y_pred:
            # Using mode_idx + 1 as a placeholder for the row index (hacky)
            rows_to_plot.append(mode_idx + 1)
            cms_to_plot.append({
                "key": mode["name"],
                "title": mode["label"],
            })

    # Create figure and gridspec dynamically
    num_rows = len(rows_to_plot)
    figsize_height = 2 + num_rows * 3  # rough estimate, tune this as needed
    fig = plt.figure(figsize=(7, figsize_height))
    gs = fig.add_gridspec(
        nrows=num_rows, ncols=4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1] * num_rows,  # uniform height ratios are simpler here
    )
    ax = []
    for row_idx in range(num_rows):
        if rows_to_plot[row_idx] == 0:
            ax.append((
                fig.add_subplot(gs[row_idx, 0]), fig.add_subplot(gs[row_idx, 1]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))
        else:
            ax.append((
                fig.add_subplot(gs[row_idx, 0:2]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))

    # Plot confusion matrices
    row_map = {original: new for new, original in enumerate(sorted(rows_to_plot))}
    for cm_info in cms_to_plot:
        key = cm_info["key"]
        title = cm_info["title"]
        mode_idx = [i for i, mode in enumerate(POOLED_MODES) if mode["name"] == key][0]
        new_row_idx = row_map[mode_idx + 1]
        plot_cm(ax=ax[new_row_idx][0], y_true=y_true[key], y_pred=y_pred[key], title_flag=title)

    # Plot each metric in a separate subplot
    for metric_name, plot_dict in metric_dict.items():
        plot_values = plot_dict["values"]
        mean = np.mean(plot_values)
        bar_kwargs = {"capsize": 5, "alpha": 0.75, "color": plot_dict["color"]}
        if isinstance(plot_values, Iterable) and len(plot_values) > 1:
            bar_kwargs["yerr"] = np.std(plot_values, ddof=1) / np.sqrt(len(plot_values))
        else:
            bar_kwargs["yerr"] = None

        # Use the mapping to get the correct new row index
        i, j = plot_dict["loc"]
        if i in row_map:
            new_i = row_map[i]
            if new_i < len(ax) and j < len(ax[new_i]):
                current_ax = ax[new_i][j]
                current_ax.bar(0, mean, **bar_kwargs)
                current_ax.set_xticks([])
                current_ax.set_ylim([0.0, plot_dict["max_y"]])
                current_ax.set_ylabel(f"[{plot_dict['unit']}]")
                current_ax.set_title(metric_name.split("\n")[0])

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
    possible_labels: list[Any] = [-1, 0, 1, 2, 3, 4, 5, 6],  # mRS in this case
) -> None:
    """
    Plot a confusion matrix given model prediction and true label values
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
    """
    Print information about available GPU(s)
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("GPU Information:\n")
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.")


def get_model_number_of_parameters(model_id: str) -> int:
    """
    Get the number of parameters in a huggingface model, using various methods
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
    """
    Get the number of parameters using heuristics from my own file-naming system
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
    """
    Compute the amount of GPU memory used at the moment by the current PID
    
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

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections.abc import Iterable
from sklearn.metrics import confusion_matrix

POOLED_MODES = [
    {"name": "all", "label": "all models", "pred_pool_mode": "concatenation"},
    {"name": "single", "label": "single model", "pred_pool_mode": "single"},
    {"name": "maj_3", "label": "maj-pooling-3", "pred_pool_mode": "majority", "num_models": 3},
    {"name": "maj_5", "label": "maj-pooling-5", "pred_pool_mode": "majority", "num_models": 5},
    {"name": "maj_10", "label": "maj-pooling-10", "pred_pool_mode": "majority", "num_models": 10},
]


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
    possible_labels: list[int] = [-1, 0, 1, 2, 3, 4, 5, 6],  # mRS in this case
    add_group_patches: bool = True,
) -> None:
    """
    Plot a confusion matrix given model prediction and true label values.
    Optionally adds red squares around specified groups of cells.
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

    # If required, add squares to highlight label groups
    if add_group_patches:
        # Group 1: no impairement (label = from 0 to 1)
        ax.add_patch(patches.Rectangle(
            xy=(0.5, 0.5), width=2, height=2, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))

        # Group 2: impairement (label = from 2 to 5)
        ax.add_patch(patches.Rectangle(
            xy=(2.5, 2.5), width=4, height=4, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))

        # Group 3: dead (label = 6)
        ax.add_patch(patches.Rectangle(
            xy=(6.5, 6.5), width=1, height=1, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))

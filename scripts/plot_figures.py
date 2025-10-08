import os
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


INPUT_DIR = "results/vllm-serve-async_guided"
OUTPUT_DIR = os.path.join(INPUT_DIR, "pooled")
OUTPUT_NAME = "pooled_results_qwen3_abiram"
PLOTTED_X_ID = "vram"
PLOTTED_Y_ID = "distance"
CASES = [
    "single model",
    "maj-pooling-3",
    "maj-pooling-5",
    # "maj-pooling-10",
    "all 5 models"
]
X_CONFIGS = {
    "vram": {"key": "VRAM param usage", "unit": "GB", "lim": None, "log": True},
    "nparams": {"key": "Number of params", "unit": "Billion", "lim": None, "log": True},
    "nbits": {"key": "Bits/param", "unit": "[]", "lim": None, "log": False},
}
Y_CONFIGS = {
    "error": {"key": "Error Rate", "unit": "%", "lim": [0.0, 1.1], "log": False},
    "distance": {"key": "Distance", "unit": "mRS unit", "lim": [0.1, 5.0], "log": True},
}
GROUP_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def _extract_metric_value(
    data_source: dict,
    key: str,
    default_val: float=None,
) -> float:
    """ Helper to extract a single metric value
    """
    item_data = data_source.get(key)
    if item_data and isinstance(item_data.get("values"), list) and item_data["values"]:
        try:
            return float(item_data["values"][0])  # Get the first element
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not parse value for key {key} from {item_data.get('values')}: {e}. Using default: {default_val}.")
    return default_val


def generate_error_rate_plot(
    result_path_group: dict[str, list],
    output_name: str,
    output_dir: str,
) -> None:
    """ Generate a single figure with subplots for Error Rate vs VRAM,
        each model group plotted with a different color within each subplot
    """
    # Plot each case data to a subplot
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    num_cols = 2
    num_rows = math.ceil(len(CASES) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows), squeeze=False)
    axes_flat = axes.flatten()
    csv_data = []
    for i, case_name in enumerate(CASES):
        for group_idx, (group_label, result_paths_in_group) in enumerate(result_path_group.items()):
            group_data_points = []
            group_color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]

            for result_path in result_paths_in_group:
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load or parse JSON file {result_path}: {e}. Skipping.")
                    continue

                extracted_data = {"model": group_label}

                for x_id in X_CONFIGS:
                    query_key = X_CONFIGS[x_id]["key"]
                    extracted_data[x_id] = _extract_metric_value(data, query_key)

                for y_id in Y_CONFIGS:
                    query_key = f"{Y_CONFIGS[y_id]['key']}\n({case_name})"
                    y_id_cased = f"{y_id} - {case_name}"
                    extracted_data[y_id_cased] = _extract_metric_value(data, query_key)

                group_data_points.append(extracted_data)

            # Scatter plot for the current group
            x_values = [dp[PLOTTED_X_ID] for dp in group_data_points]
            plotted_y_id_cased = f"{PLOTTED_Y_ID} - {case_name}"
            y_values = [dp[plotted_y_id_cased] for dp in group_data_points]
            sizes = [200 * dp["nbits"] / 16 for dp in group_data_points]
            axes_flat[i].scatter(
                x_values, y_values, color=group_color, label=group_label,
                marker="o", alpha=0.7, s=sizes,
            )

            # Record data for pooled json file
            if len(group_data_points) > 0:
                csv_data.extend(group_data_points)

        # Configure subplot
        x_label = f"{X_CONFIGS[PLOTTED_X_ID]['key']} [{X_CONFIGS[PLOTTED_X_ID]['unit']}]"
        y_label = f"{Y_CONFIGS[PLOTTED_Y_ID]['key']} [{Y_CONFIGS[PLOTTED_Y_ID]['unit']}]"
        if X_CONFIGS[PLOTTED_X_ID]['log']: axes_flat[i].set_xscale('log')
        if Y_CONFIGS[PLOTTED_Y_ID]['log']: axes_flat[i].set_yscale('log')
        axes_flat[i].set_xlabel(x_label, fontsize=12)
        axes_flat[i].set_ylabel(y_label, fontsize=12)
        axes_flat[i].set_ylim(Y_CONFIGS[PLOTTED_Y_ID]["lim"])
        axes_flat[i].tick_params(axis="y", labelsize=10)
        axes_flat[i].tick_params(axis="x", labelsize=10)
        axes_flat[i].grid(True, linestyle="--", alpha=0.6)
        axes_flat[i].set_title(f"Prediction with {case_name}", fontsize=14, pad=10)
        axes_flat[i].legend(loc="upper right", fontsize=12, fancybox=True, ncol=1)

    # Save the pooled results figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_full_path = os.path.join(output_dir, output_name + ".png")
    plt.savefig(plot_full_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Combined plot saved: {plot_full_path}")

    # Save the pooled data to a csv file using pandas
    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.groupby(["model", "vram", "nparams", "nbits"]).first().reset_index()
    csv_full_path = os.path.join(output_dir, output_name + ".csv")
    csv_df.to_csv(csv_full_path, index=False)
    print(f"Pooled data saved: {csv_full_path}")

    return plot_full_path, csv_full_path


def fit_error_model_lme(
    df_input: pd.DataFrame,
    dependent_variable: str="error - single model",
    fixed_effects: list[str]=["vram", "nbits"],
    random_intercept_group: str="model",
):
    """ Fits a linear mixed-effects model to the provided dataframe
        The model is defined as: error_single_model ~ vram + nbits + (1|model)
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df_input.copy()
    df.rename(columns={dependent_variable: "target"}, inplace=True)

    # Define interaction model - 'vram * nbits' expands to 'vram + nbits + vram:nbits'
    model_formula = f"target ~ {fixed_effects[0]} * {fixed_effects[1]}"
    md = smf.mixedlm(model_formula, df, groups=df[random_intercept_group])
    
    # Returns fitted mixed-effects model (call ".summary()" for printing results)
    return md.fit()


if __name__ == "__main__":

    # Define paths to all plotted models
    result_path_group = {

        "Qwen3-0.6B": [
            # "unsloth/Qwen3-0.6B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-0.6B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q8_0.json",
            "Qwen/Qwen3-0.6B-FP8.json",
        ],

        "Qwen3-1.7B": [
            # "unsloth/Qwen3-1.7B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-1.7B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q8_0.json",
            "Qwen/Qwen3-1.7B-FP8.json",
        ],

        "Qwen3-4B": [
            # "unsloth/Qwen3-4B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-4B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q8_0.json",
            "Qwen/Qwen3-4B-AWQ.json",
            "Qwen/Qwen3-4B-FP8.json",
        ],

        "Qwen3-8B": [
            # "unsloth/Qwen3-8B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-8B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q8_0.json",
            "Qwen/Qwen3-8B-AWQ.json",
            "Qwen/Qwen3-8B-FP8.json",
        ],

        "Qwen3-14B": [
            # "unsloth/Qwen3-14B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-14B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q8_0.json",
            "Qwen/Qwen3-14B-AWQ.json",
            "Qwen/Qwen3-14B-FP8.json",
        ],

        "Qwen3-32B": [
            # "unsloth/Qwen3-32B-GGUF-IQ1_M.json",
            "unsloth/Qwen3-32B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q8_0.json",
            "Qwen/Qwen3-32B-AWQ.json",
            "Qwen/Qwen3-32B-FP8.json",
        ],

    }

    # Prepend input directory to all result paths
    result_path_group = {
        group: [os.path.join(INPUT_DIR, path) for path in paths]
        for group, paths in result_path_group.items()
    }

    # Pool results and plot them
    output_png_path, output_csv_path = generate_error_rate_plot(
        result_path_group,
        output_name=OUTPUT_NAME,
        output_dir=OUTPUT_DIR,
    )

    # # Identify statistical patterns using linear mixed-effects models
    # lme_results = fit_error_model_lme(pd.read_csv(output_csv_path))
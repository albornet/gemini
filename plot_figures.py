import os
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


X_CONFIGS = {
    "vram": {"key": "VRAM param usage", "unit": "GB", "lim": None},
    "nparams": {"key": "Number of params", "unit": "Billion", "lim": None},
    "nbits": {"key": "Bits/param", "unit": "[]", "lim": None},
}
Y_CONFIGS = {
    "error": {"key": "Error Rate", "unit": "%", "lim": [0.0, 1.1]},
    "distance": {"key": "Distance", "unit": "%", "lim": [0.0, 4.0]},
}
PLOTTED_X_ID = "vram"
PLOTTED_Y_ID = "distance"


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
    plot_name: str,
    output_dir: str,
) -> None:
    """ Generate a single figure with subplots for Error Rate vs VRAM,
        each model group plotted with a different color within each subplot
    """
    # Initialization and static variables
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    cases = ["single model", "maj-pooling-3", "maj-pooling-5", "maj-pooling-10"]
    group_colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]

    # Plot each case data to a subplot
    num_cols = 2
    num_rows = math.ceil(len(cases) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows), squeeze=False)
    axes_flat = axes.flatten()
    csv_data = []
    for i, case_name in enumerate(cases):
        for group_idx, (group_label, result_paths_in_group) in enumerate(result_path_group.items()):
            group_data_points = []
            group_color = group_colors[group_idx % len(group_colors)]

            for result_path in result_paths_in_group:
                with open(result_path, "r") as f:
                    data = json.load(f)
                
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
            csv_data.extend(group_data_points)

        # Configure subplot
        x_label = f"{X_CONFIGS[PLOTTED_X_ID]['key']} ({X_CONFIGS[PLOTTED_X_ID]['unit']})"
        y_label = f"{Y_CONFIGS[PLOTTED_Y_ID]['key']} ({Y_CONFIGS[PLOTTED_Y_ID]['unit']})"
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
    plot_full_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_full_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined plot saved: {plot_full_path}")

    # Save the pooled data to a csv file using pandas
    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.groupby(["model", "vram", "nparams", "nbits"]).first().reset_index()
    csv_full_path = os.path.join(output_dir, plot_name + ".csv")
    csv_df.to_csv(csv_full_path, index=False)
    print(f"Pooled data saved: {csv_full_path}")


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
    llama_result_path_group = {
        "Llama-3.2-1B-Instruct": [
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-Q8_0.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF-FP16.json",
        ],
        "Llama-3.2-3B-Instruct": [
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-Q8_0.json",
            "results/vllm_guided/MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF-FP16.json",
        ],
        "Llama-3.1-8B-Instruct": [
            "results/vllm_guided/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF-Q2_K.json",
            "results/vllm_guided/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF-Q3_K_M.json",
            "results/vllm_guided/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF-Q4_K_M.json",
            "results/vllm_guided/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF-Q5_K_M.json",
            "results/vllm_guided/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF-Q6_K.json",
        ],
    }

    qwen_result_path_group = {
        "Qwen2.5-0.5B-Instruct": [
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-Q8_0.json",
            "results/vllm_guided/bartowski/Qwen2.5-0.5B-Instruct-GGUF-F16.json",
        ],

        "Qwen2.5-1.5B-Instruct": [
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-Q8_0.json",
            "results/vllm_guided/bartowski/Qwen2.5-1.5B-Instruct-GGUF-F16.json",
        ],

        "Qwen2.5-3B-Instruct": [
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-Q8_0.json",
            "results/vllm_guided/bartowski/Qwen2.5-3B-Instruct-GGUF-F16.json",
        ],

        "Qwen2.5-7B-Instruct": [
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q3_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q4_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q5_K_M.json",
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q6_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-7B-Instruct-GGUF-Q8_0.json",
        ],

        "Qwen2.5-14B-Instruct": [
            "results/vllm_guided/bartowski/Qwen2.5-14B-Instruct-GGUF-Q2_K.json",
            "results/vllm_guided/bartowski/Qwen2.5-14B-Instruct-GGUF-Q3_K_M.json",
        ],
    }

    output_dir = "pooled_result"
    generate_error_rate_plot(llama_result_path_group, plot_name="llama_pooled_results", output_dir=output_dir)
    generate_error_rate_plot(qwen_result_path_group, plot_name="qwen_pooled_results", output_dir=output_dir)
    llama_lme_results = fit_error_model_lme(pd.read_csv(os.path.join(output_dir, "llama_pooled_results.csv")))
    qwen_lme_results = fit_error_model_lme(pd.read_csv(os.path.join(output_dir, "qwen_pooled_results.csv")))
    import ipdb; ipdb.set_trace()
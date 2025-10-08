#!/bin/bash

# Configuration
CONFIG_FILE="./configs/model_config.yaml"
INFERENCE_BACKEND="vllm-serve-async"
GPU_MEM_UTIL="0.80"
MAX_CONCURRENT_INFS="64"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'"
    exit 1
fi

# Models and quantizations to test (GGUF)
MODEL_PATHS=(
    # "unsloth/Qwen3-0.6B-GGUF"
    "unsloth/Qwen3-1.7B-GGUF"
    "unsloth/Qwen3-4B-GGUF"
    "unsloth/Qwen3-8B-GGUF"
)
QUANT_SCHEMES=(
    "Q2_K_XL"
    "Q3_K_XL"
    "Q4_K_XL"
    "Q5_K_XL"
    "Q6_K_XL"
    "Q8_0"
    "Q8_K_XL"
)

# # Models and quantizations to test (AWQ/FP8)
# MODEL_PATHS=(
#     # Orion-zhen/Qwen3-0.6B-AWQ
#     Qwen/Qwen3-0.6B-FP8
#     # Orion-zhen/Qwen3-1.7B-AWQ
#     Qwen/Qwen3-1.7B-FP8
#     Qwen/Qwen3-4B-AWQ
#     Qwen/Qwen3-4B-FP8
#     Qwen/Qwen3-8B-AWQ
#     Qwen/Qwen3-8B-FP8
# )
# QUANT_SCHEMES=(
#     "no-quant-scheme"
# )

# Main loop
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for QUANT_SCHEME in "${QUANT_SCHEMES[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Running benchmark for:"
        echo "   Model: $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"
        echo "--------------------------------------------------------"

        # Set GPU_MEM_UTIL based on model and quantization
        if [[ "$MODEL_PATH" == "unsloth/Qwen3-8B-GGUF" && ( "$QUANT_SCHEME" == "Q8_0" || "$QUANT_SCHEME" == "Q8_K_XL" ) ]]; then
            GPU_MEM_UTIL="0.90"
        else
            GPU_MEM_UTIL="0.80"  # reset to default for other combinations
        fi

        # Update the model configuration file
        sed -i "s|^model_path:.*|model_path: $MODEL_PATH|" "$CONFIG_FILE"
        sed -i "s|^quant_scheme:.*|quant_scheme: $QUANT_SCHEME|" "$CONFIG_FILE"
        sed -i "s|^inference_backend:.*|inference_backend: $INFERENCE_BACKEND|" "$CONFIG_FILE"
        sed -i "s|^gpu_memory_utilization:.*|gpu_memory_utilization: $GPU_MEM_UTIL|" "$CONFIG_FILE"
        sed -i "s|^max_concurrent_inferences:.*|max_concurrent_infs: $MAX_CONCURRENT_INFS|" "$CONFIG_FILE"
        echo "Updated '$CONFIG_FILE' for the current run."

        # Run the benchmark with updated configuration
        python -m scripts.run_benchmark \
            --encrypted-data-path "./data/data_2025/processed/dataset.encrypted.csv" \
            --curated-data-path "./data/data_2024/processed/dataset.csv" \
            --remote-env-path "/home/shares/ds4dh/gemini_project/gemini/.env" \
            --key-name "GEMINI" \
            --hostname "login1.baobab.hpc.unige.ch" \
            --username "borneta" \
            --model-config-path "./configs/model_config.yaml" \
            --data-config-path "./configs/data_config.yaml" \
            --prompt-config-path "./configs/prompt_config.yaml" \
            --output-config-path "./configs/output_config.yaml"

        # Check the exit code of the benchmark script
        if [ $? -ne 0 ]; then
            echo "Error: Benchmark failed for $MODEL_PATH ($QUANT_SCHEME). Aborting."
            exit 1
        fi

        echo "Benchmark finished for $MODEL_PATH ($QUANT_SCHEME)."
        echo ""

        # Small delay between runs
        sleep 1
    done
done

echo "All benchmarks completed successfully!"
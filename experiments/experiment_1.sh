#!/bin/bash

PARTITION="private-teodoro-gpu"  # "private-teodoro-gpu" // "shared-gpu"
TIME_TO_RUN="0-02:00:00"
DEFAULT_NUM_GPUS=1

CONFIG_FILE="./configs/model_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'"
    exit 1
fi
INFERENCE_BACKEND="vllm-serve-async"

MODEL_PATHS=(
    "unsloth/Qwen3-0.6B-GGUF"
    "unsloth/Qwen3-1.7B-GGUF"
    "unsloth/Qwen3-4B-GGUF"
    "unsloth/Qwen3-8B-GGUF"
    "unsloth/Qwen3-14B-GGUF"
    "unsloth/Qwen3-32B-GGUF"
)

QUANT_SCHEMES=(
    "IQ1_M"
    "Q2_K_XL"
    "Q3_K_XL"
    "Q4_K_XL"
    "Q5_K_XL"
    "Q6_K_XL"
    "Q8_K_XL"
)

declare -A GPU_REQUIREMENTS
GPU_REQUIREMENTS["unsloth/Qwen3-14B-GGUF:Q8_K_XL"]="2"
GPU_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q4_K_XL"]="2"
GPU_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q5_K_XL"]="2"
GPU_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q6_K_XL"]="2"
GPU_REQUIREMENTS["unsloth/Qwen3-32B-GGUF:Q8_K_XL"]="3"

# Iterate over each model and each quantization mode
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for QUANT_SCHEME in "${QUANT_SCHEMES[@]}"; do
        echo "--------------------------------------------------------"
        echo "   Preparing benchmark for:"
        echo "   Model: $MODEL_PATH"
        echo "   Quantization: $QUANT_SCHEME"

        # Select the number of GPU to use for that combination
        KEY="${MODEL_PATH}:${QUANT_SCHEME}"
        NUM_GPUS=$DEFAULT_NUM_GPUS
        if [[ -v "GPU_REQUIREMENTS[$KEY]" ]]; then
            NUM_GPUS=${GPU_REQUIREMENTS[$KEY]}
            echo "   Found custom config: Requesting $NUM_GPUS GPU(s)."
        else
            echo "   Using default config: Requesting $NUM_GPUS GPU(s)."
        fi
        echo "--------------------------------------------------------"

        # Modify the YAML file using sed
        # The '-i' flag modifies the file in-place.
        # We use '|' as the delimiter for 's' command to avoid conflicts with '/' in the MODEL_PATH.
        sed -i "s|^model_path:.*|model_path: $MODEL_PATH|" "$CONFIG_FILE"
        sed -i "s|^quant_scheme:.*|quant_scheme: $QUANT_SCHEME|" "$CONFIG_FILE"
        sed -i "s|^inference_backend:.*|inference_backend: $INFERENCE_BACKEND|" "$CONFIG_FILE"
        echo "Updated '$CONFIG_FILE'."

        # Run the benchmark script
        echo "Running benchmark script..."
        ./scripts/run_benchmark.sh -g "$NUM_GPUS" -t $TIME_TO_RUN -p $PARTITION
        echo "Benchmark job sent for $MODEL_PATH ($QUANT_SCHEME)."
        echo ""

        # Wait for 1 second to prevent a race condition
        sleep 1
    done
done
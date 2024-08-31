class Config:
    DEBUG = False
    RESULT_DIR = "results"
    DATASET_PATH = "./data/dataset.csv"
    RUNS = [
        # Models using classic transformers pipeline
        # {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "quantize_mode": "bits_and_bytes"},  # quantized at runtime
        # {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "quantize_mode": None},  # "full" version
        # {"model_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"},
        # {"model_id": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"},
        
        # Models using Llama-cpp-python with GGUF quantization (small, general)
        {"model_id": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ1_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ2_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ3_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf"},
        
        # Models using Llama-cpp-python with GGUF quantization (small, specialized)
        {"model_id": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ1_M.gguf"},
        {"model_id": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ2_M.gguf"},
        {"model_id": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ3_M.gguf"},
        {"model_id": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-Q4_K_M.gguf"},
        # {"model_id": "mradermacher/OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.Q2_K.gguf"},
        # {"model_id": "mradermacher/OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.IQ3_M.gguf"},
        # {"model_id": "mradermacher/OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.Q4_K_M.gguf"},
        
        # Models using Llama-cpp-python with GGUF quantization (large, general)
        {"model_id": "mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ1_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ2_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ3_M.gguf"},
        {"model_id": "mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ4_XS.gguf"},
        
        # Models using Llama-cpp-python with GGUF quantization (large, specialized)
        {"model_id": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ1_M.gguf"},
        {"model_id": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ2_M.gguf"},
        {"model_id": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ3_M.gguf"},
        {"model_id": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ4_XS.gguf"},
    ]
    AUTO_GPTQ_MODELS = [
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",
    ]
    MAX_INPUT_LENGTH = 4096  # approximately the maximum of tokens in a "lettre de sortie" (?)

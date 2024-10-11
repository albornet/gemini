class Config:
    DEBUG = True
    RESULT_DIR = "results"
    DATASET_PATH = "./data/dataset.csv"
    USE_FLASH_ATTENTION = True
    MAX_CONTEXT_LENGTH = 6144  # maximum tokens in a "lettre de sortie"
    RUN_DICT = {
        
        # Models requiring a very small GPU (up to ~1G)
        "very_small": [
            # Very small models!
            {"repo_name": "mradermacher", "model_id": "Llama-3.2-1B-Instruct-i1-GGUF", "quantize_mode": "Llama-3.2-1B-Instruct.i1-IQ1_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Llama-3.2-1B-Instruct-i1-GGUF", "quantize_mode": "Llama-3.2-1B-Instruct.i1-IQ2_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Llama-3.2-1B-Instruct-i1-GGUF", "quantize_mode": "Llama-3.2-1B-Instruct.i1-IQ3_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Llama-3.2-1B-Instruct-i1-GGUF", "quantize_mode": "Llama-3.2-1B-Instruct.i1-Q4_K_M.gguf"},
        ],
        
        # Models requiring a small GPU (up to ~10G)
        "small": [
            # General Llama-8B models
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ1_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ2_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-IQ3_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf"},
            
            # Specialized Llama-8B models
            {"repo_name": "mradermacher", "model_id": "Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ1_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ2_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-IQ3_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Bio-Medical-Llama-3.1-8B-i1-GGUF", "quantize_mode": "Bio-Medical-Llama-3.1-8B.i1-Q4_K_M.gguf"},
            
            # # Other specialized Llama-8B models
            # {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.Q2_K.gguf"},
            # {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.IQ3_M.gguf"},
            # {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-8B-GGUF", "quantize_mode": "OpenBioLLM-Llama3-8B.Q4_K_M.gguf"},
            
            # # Mixed Llama-8B models
            # {"repo_name": "mradermacher", "model_id": "Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quantize_mode": "Llama3-Instruct-OpenBioLLM-8B-merged.Q2_K.gguf"},
            # {"repo_name": "mradermacher", "model_id": "Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quantize_mode": "Llama3-Instruct-OpenBioLLM-8B-merged.IQ3_M.gguf"},
            # {"repo_name": "mradermacher", "model_id": "Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quantize_mode": "Llama3-Instruct-OpenBioLLM-8B-merged.Q4_K_M.gguf"},
        ],
        
        # Models requiring a big GPU (up to ~45G)
        "big": [
            # General Llama-70B models
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ1_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ2_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ3_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-IQ4_XS.gguf"},
            # {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-Q4_K_S.gguf"},  # 40.4G /!\ -> maybe fast-attn makes it work?
            # {"repo_name": "mradermacher", "model_id": "Meta-Llama-3.1-70B-Instruct-i1-GGUF", "quantize_mode": "Meta-Llama-3.1-70B-Instruct.i1-Q4_K_M.gguf"},  # 42.6G /!\ -> maybe fast-attn makes it work?
            
            # Specialized Llama-8B models
            {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ1_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ2_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ3_M.gguf"},
            {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-IQ4_XS.gguf"},
            # {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-Q4_K_S.gguf"},  # 40.4G /!\ -> maybe fast-attn makes it work?
            # {"repo_name": "mradermacher", "model_id": "OpenBioLLM-Llama3-70B-i1-GGUF", "quantize_mode": "OpenBioLLM-Llama3-70B.i1-Q4_K_M.gguf"},  # 42.6G /!\ -> maybe fast-attn makes it work?
        ],
        
        # INFERENCE CONSIDERATIONS
        # WE CAN RUN ALL LLAMA-8B WITH A "BASIC" NVIDIA-GEFORCE-RTX-3090 (24G) GPU (1'000$ - 1'250$)
        # WE CAN RUN DECENT VERSIONS OF LLAMA-70B WITH A MORE EXPENSIVE TESLA V-100 (40G) GPU (9'000$ - 13'000$)
        # NEED TO CHECK WITH FLASH-ATTN
        
        # TRAINING CONSIDERATIONS
        # TO FINE-TUNE LLAMA-70B MODEL WITH LORA, WE NEED APPROXIMATELY 192G V-RAM
        # ONE 40G TESLA V-100 IS 9'000$ - 13'000$ => 5 OF THESE WOULD BE 45'000$ - 65'000$ FOR 200G V-RAM (QUITE FAST)
        # ONE 24G RTX 3090 IS 1'000$ - 1'250$ => 8 OF THESE WOULD BE 8'000$ - 10'000$ FOR 192G V-RAM (QUITE SLOW)
    }
    

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from llama_cpp import Llama
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from warnings import warn
from src.utils.run_utils import load_config

cfg = load_config()


def get_model_and_tokenizer(
    model_path: str,
    quant_method: str,
    quant_scheme: str|None=None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """ Create an LLM-based inference generator for solving a task
    
    Args:
        model_path (str): reference string to load model from huggingface
        quant_method (str): model file format (normal, bitsandbytes, awq, tqdm, gguf, etc.)
    """
    tokenizer = None
    match cfg.INFERENCE_BACKEND:
        
        # Using vLLM backend
        case "vllm":
            model_args = {
                "trust_remote_code": True,
                "max_model_len": cfg.MAX_CONTEXT_LENGTH,
            }
            if quant_method == "bnb":
                raise ValueError(f"vLLM does not support format {quant_method}")
            elif quant_method == "gguf":
                model_file_path = download_gguf_by_quant(model_path, quant_scheme)
                tokenizer_path = get_tokenizer_name(model_path)
                model_args.update({"model": model_file_path, "tokenizer": tokenizer_path})
            else:
                model_args.update({"model": model_path, "quantization": quant_method})
            model = LLM(**model_args)
        
        # Using Llama-cpp backend
        case "llama-cpp":
            if quant_method != "gguf":
                raise ValueError(f"Llama-cpp does not support format {quant_method}")
            model = Llama.from_pretrained(
                repo_id=model_path,
                filename=f"*{quant_scheme}*.gguf",
                n_gpu_layers=-1,
                n_ctx=cfg.MAX_CONTEXT_LENGTH,
                flash_attn=cfg.USE_FLASH_ATTENTION,
                verbose=False,
            )

        # Using HuggingFace backend
        case "hf":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                attn_implementation="flash_attention_2" if cfg.USE_FLASH_ATTENTION else None,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer


def get_tokenizer_name(
    model_id: str,
    chat_template_required: bool=True,
) -> str:
    """ Identify base model from which any model was quantized, in order to load
        the correct tokenizer
    """
    # Look for base model in the "cardData" (where model tree info is stored)
    api = HfApi()
    model_info = api.model_info(model_id)
    card_data = model_info.card_data or {}
    tokenizer_name = card_data.get("base_model")
    
    # Alternatively, inspect tags or siblings
    if not tokenizer_name:
        for tag in model_info.tags:
            if "base_model:" in tag:
                tokenizer_name = tag.split(":")[1]
                break
    
    # Check for chat template, may fall back on Llama-3.2-3B-Instruct (most common)
    if chat_template_required and tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.chat_template is None:
                raise ValueError("chat_template missing")
        except Exception as e:
            warn(
                f"The tokenizer '{tokenizer_name}' does not support chat templating "
                f"(reason: {e}). Falling back to 'meta-llama/Llama-3.2-3B-Instruct'."
            )
            tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"

    return tokenizer_name




def download_gguf_by_quant(model_id: str, quant_scheme: str) -> str:
    """ Download the first matching GGUF file in a model repository
    """
    quant_scheme_lower = quant_scheme.lower()
    files = list_repo_files(model_id)
    for file in files:
        file_lower = file.lower()
        if file_lower.endswith(".gguf") and quant_scheme_lower in file_lower:
            return hf_hub_download(repo_id=model_id, filename=file)
        
    raise FileNotFoundError(f"No GGUF file found with quantization scheme '{quant_scheme}' in repo '{model_id}'")

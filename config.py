from argparse import ArgumentParser, Namespace
from typing import Union
from transformers import AutoTokenizer


class Config:
    RESULT_DIR = "results"
    DATASET_PATH = "./data/dataset.csv"
    N_INFERENCE_REPEATS = 10  # number of times each inference is repeated for benchmark
    MAX_CONTEXT_LENGTH = 6144  # maximum tokens in a "lettre de sortie", useful for llama-cpp
    MAX_GENERATED_TOKENS = 512
    TOP_P = 0.95
    TEMPERATURE = 0.80
    USE_FLASH_ATTENTION = True
    INFERENCE_BACKEND = "vllm"  # hf, vllm, llama-cpp
    GGUF_QUANT_SCHEME = "Q4_K_M"
    RUN_DICT = {
        
        # Models using up to ~4G of GPU memory
        "very_small": [

            # General domain
            {"model_path": "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "quant": "awq"},
            {"model_path": "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "quant": "awq"},
            {"model_path": "Qwen/Qwen2.5-3B-Instruct-AWQ", "quant": "awq"},
            {"model_path": "mradermacher/Llama-3.2-1B-Instruct-i1-GGUF", "quant": "gguf"},
            {"model_path": "mradermacher/Llama-3.2-3B-Instruct-i1-GGUF", "quant": "gguf"},

            # Adapted to the biomedical domain
            {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf"},

        ],

        # Models using up to ~12G of GPU memory
        "small": [

            # General domain model
            # {"model_path": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quant": "awq"},
            # {"model_path": "Qwen/Qwen2.5-14B-Instruct-AWQ", "quant": "awq"},  # Only 2k tokens max?! I thought 128k, why?
            {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf"},

            # Adapted to the biomedical domain
            {"model_path": "PrunaAI/ContactDoctor-Bio-Medical-Llama-3-8B-AWQ-4bit-smashed", "quant": "awq"},
            {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf"},
            {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-GGUF", "quant": "gguf"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-7B-GGUF", "quant": "gguf"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-QWEN2.5-i1-GGUF", "quant": "gguf"},

        ],

        # Models using more than ~12G of GPU memory
        "large": [

            # General domain models
            {"model_path": "Qwen/Qwen2.5-32B-Instruct-AWQ", "quant": "awq"},
            {"model_path": "Qwen/Qwen2.5-72B-Instruct-AWQ", "quant": "awq"},
            {"model_path": "mradermacher/Llama-3.3-70B-Instruct-i1-GGUF", "quant": "gguf"},

            # Adapted to the biomedical domain
            {"model_path": "TitanML/Llama3-OpenBioLLM-70B-AWQ-4bit", "quant": "awq"},
            {"model_path": "mradermacher/DeepSeek-R1-Distill-Qwen-32B-Medical-i1-i1-GGUF", "quant": "gguf"},
            {"model_path": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quant": "gguf"},
            {"model_path": "mradermacher/Llama-3-70B-UltraMedical-i1-GGUF", "quant": "gguf"},

        ],
    }


def build_prompt(input_text: str) -> list[dict[str, str]]:
    """ Build a message list for prompting a large language model for the task at hand
    
    Args:
        input_text (str): core input text for the current sample

    Returns:
        list[dict[str, str]]: messages to prompt a large language model
    """
    system_prompt = (
        "Tu es un neuro-chirurgien et ta tâche est d’identifier le score sur l’Échelle de Rankin Modifiée (mRS) du patient "
        "en fonction de son état après sa sortie, tel qu'il est décrit dans la lettre de sortie rédigée par un autre médecin.\n"
        "L'Échelle de Rankin Modifiée (mRS) est utilisée pour mesurer le degré d'incapacité chez les patients ayant subi un accident vasculaire cérébral (AVC) :\n"
        "0 : Aucun symptôme\n"
        "1 : Aucune incapacité significative malgré des symptômes ; capable d'effectuer toutes les tâches et activités habituelles\n"
        "2 : Légère incapacité ; incapable d'effectuer toutes les activités antérieures, mais capable de s'occuper de ses propres affaires sans assistance\n"
        "3 : Incapacité modérée ; nécessitant une certaine aide, mais capable de marcher sans assistance\n"
        "4 : Incapacité modérément sévère ; incapable de marcher sans assistance et incapable de s'occuper de ses besoins corporels sans assistance\n"
        "5 : Incapacité sévère ; alité, incontinent et nécessitant des soins infirmiers constants et une attention continue\n"
        "6 : Décédé\n"
    )
    intro_text = "Voici la lettre du patient :"
    output_specification = (
        "Pour rédiger ta réponse :\n"
        "1. Explique de manière détaillée les étapes de ton raisonnement en t’appuyant sur "
        "les informations pertinentes de la lettre de sortie.\n"
        "2. À la fin de ta réponse, sur une nouvelle ligne, indique uniquement le score mRS "
        "prédit pour le patient, au format exact suivant :\n\n"
        "mRS : <nombre>\n"
    )
    user_prompt = "\n".join((intro_text, input_text, output_specification))
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    
    return messages


def build_prompts(
    sample: dict[str, str],
    tokenizer: AutoTokenizer,
) -> dict[str, Union[list, str]]:
    """ Take in a data sample, build huggingface style's messages,
        add tokenize associated prompt
    """
    messages = build_prompt(sample["input_text"])
    prompt = None
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return {"messages": messages, "prompt": prompt}


def parse_script_args() -> Namespace:
    """ Parse and validate command line arguments
    
    Returns:
        Namespace containing:
        - runtype: str (very_small, small, big, all)
        - debug: bool
        - plot_only: bool
    """
    parser = ArgumentParser(description="Benchmark LLMs on healthcare tasks")
    
    # Run type configuration
    parser.add_argument(
        "-t", "--runtype",
        default="very_small",
        choices=["very_small", "small", "large", "all"],
        help="Scope of benchmark: very_small, small, big, or all"
    )
    
    # Debug mode flag
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    # Plot mode flag
    parser.add_argument(
        "-p", "--plot_only",
        action="store_true",
        help="Enable plot mode (to re-plot saved results)"
    )
    
    return parser.parse_args()
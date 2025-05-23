from argparse import ArgumentParser, Namespace
from typing import Union, Annotated
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm.sampling_params import GuidedDecodingParams


class Config:
    RESULT_DIR = "results"
    DATASET_PATH = "./data/dataset.csv"
    N_INFERENCE_REPEATS = 10  # number of times each inference is repeated for benchmark
    MAX_CONTEXT_LENGTH = 6144  # maximum tokens in a "lettre de sortie" is 6144
    MAX_GENERATED_TOKENS = 512
    TOP_P = 0.95
    TEMPERATURE = 0.80
    USE_FLASH_ATTENTION = True
    INFERENCE_BACKEND = "vllm"  # hf, vllm, llama-cpp
    VLLM_USE_OUTPUT_GUIDE = True
    RUN_DICT = {

        # = already run
        # # = can't be run (e.g., specific quantization doesn't exist)
        
        # Models using up to ~4G of GPU memory
        "very_small": [

            ##################
            # General domain #
            ##################

            # Qwen2.5-0.5B-Instruct
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},

            # Qwen2.5-1.5B-Instruct
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},

            # Qwen2.5-3B-Instruct
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            # Llama-3.2-1B-Instruct (actually quantized with i1)
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "FP16"},
            
            # Llama-3.2-3B-Instruct (actually quantized with i1)
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # {"model_path": "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "FP16"},

            ####################################
            # Adapted to the biomedical domain #
            ####################################

            # Deepseek-R1-Medical-COT-Qwen-1.5 (no information about from where it was finetuned)
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # {"model_path": "mradermacher/DeepSeek-R1-Medical-COT-Qwen-1.5-GGUF", "quant": "gguf", "quant_scheme": "F16"},

            # LLAMA3-3B-Medical-COT (/!\ -> finetuned from Llama-3.2-1B-Instruct /!\, i.e., not 3B)
            # {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # # [doesn't exist] {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # # [doesn't exist] {"model_path": "mradermacher/LLAMA3-3B-Medical-COT-i1-GGUF", "quant": "gguf", "quant_scheme": "F16"},

            # ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF (finetuned from Llama-3.2-1B-Instruct)
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # {"model_path": "DevQuasar/ContactDoctor.Bio-Medical-Llama-3-2-1B-CoT-012025-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
        ],

        # Models using up to ~12G of GPU memory
        "small": [
            
            ##################
            # General domain #
            ##################

            # Qwen2.5-7B-Instruct
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # {"model_path": "bartowski/Qwen2.5-7B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},  # I think it would go OOM for F16
            
            # Qwen2.5-14B-Instruct (quantizations I can run on my 12G GPU)
            # {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},  # this one goes OOM with Q4_K_M
            # {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},  # this one goes OOM on my 12GB GPU with Q4_K_M
            
            # Llama-3.1-8B-Instruct
            # {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # # [doesn't exist] {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # # [doesn't exist] {"model_path": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            ####################################
            # Adapted to the biomedical domain #
            ####################################

            # Llama3-Instruct-OpenBioLLM-8B-merged -> OpenBioLLM-8B merged with Llama-3-8B-Instruct chat capabilities
            # {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},  # really good one
            # {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # # # [doesn't exist] {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # # # [doesn't exist] {"model_path": "mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            # Bio-Medical-Llama (built on top of Llama-3.1-8B-Instruct)
            # {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # # # [doesn't exist] {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # # # [doesn't exist] {"model_path": "mradermacher/Bio-Medical-Llama-3.1-8B-i1-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            # Llama-ChatDoctor (built on top of Llama-3-8B-Instruct)
            # {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            # {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            # {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            # {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            # # # [doesn't exist] {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            # # # [doesn't exist] {"model_path": "mradermacher/Llama-chatDoctor-i1-GGUF", "quant": "gguf", "quant_scheme": "F16"},
        
        ],

        # Models using more than ~12G of GPU memory
        "large": [

            # General domain models
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-14B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-32B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q2_K"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q3_K_M"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q5_K_M"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q6_K"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "Q8_0"},
            {"model_path": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "gguf", "quant_scheme": "F16"},
            
            # # Adapted to the biomedical domain
            # {"model_path": "TitanML/Llama3-OpenBioLLM-70B-AWQ-4bit", "quant": "awq"},
            # {"model_path": "mradermacher/DeepSeek-R1-Distill-Qwen-32B-Medical-i1-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/OpenBioLLM-Llama3-70B-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},
            # {"model_path": "mradermacher/Llama-3-70B-UltraMedical-i1-GGUF", "quant": "gguf", "quant_scheme": "Q4_K_M"},

        ],
    }


def build_prompt(input_text: str) -> list[dict[str, str]]:
    """ Build a message list for prompting a large language model for the task at hand

    Args:
        input_text (str): core input text for the current sample

    Returns:
        list[dict[str, str]]: messages to prompt a large language model
    """
    task_description = (
        "Tu es un neuro-chirurgien et ta tâche est d’identifier le score sur l’Échelle de Rankin Modifiée (mRS) du patient"
        "en fonction de son état après sa sortie, tel qu'il est décrit dans la lettre de sortie rédigée par un autre médecin."
    )
    mrs_description = (
        "L'Échelle de Rankin Modifiée (mRS) est utilisée pour mesurer le degré d'incapacité chez les patients ayant subi un accident vasculaire cérébral (AVC) :\n"
        "0 : Aucun symptôme\n"
        "1 : Aucune incapacité significative malgré des symptômes ; capable d'effectuer toutes les tâches et activités habituelles\n"
        "2 : Légère incapacité ; incapable d'effectuer toutes les activités antérieures, mais capable de s'occuper de ses propres affaires sans assistance\n"
        "3 : Incapacité modérée ; nécessitant une certaine aide, mais capable de marcher sans assistance\n"
        "4 : Incapacité modérément sévère ; incapable de marcher sans assistance et incapable de s'occuper de ses besoins corporels sans assistance\n"
        "5 : Incapacité sévère ; alité, incontinent et nécessitant des soins infirmiers constants et une attention continue\n"
        "6 : Décédé"
    )
    output_specs = (
        "Ta réponse doit être JSON conforme au format suivant :\n\n"
        "{\n"
        '  "reasoning": "<comment tu as déterminé le mRS>",\n'
        '  "mRS": <nombre entre 0 et 6>\n'
        "}\n\n"
    )
    # system_prompt = f"{task_description}\n{output_specs}"
    system_prompt = f"{task_description}\n{mrs_description}\n{output_specs}"
    user_prompt = (
        "Voici la lettre de sortie du patient:\n"
        "DEBUT DE LA LETTRE DE SORTIE :\n"
        f"{input_text}\n"
        "FIN DE LA LETTRE DE SORTIE\n"
    )
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


def get_output_guide() -> GuidedDecodingParams:
    """ Define a formatted output guide to constraint the LLM when generating tokens
    """
    class PatientInfoSchema(BaseModel):
        reasoning: str = Field(
            ...,
            max_length=int(max(Config.MAX_GENERATED_TOKENS * 0.9, Config.MAX_GENERATED_TOKENS - 100)),
            description="Le raisonnement du modèle pour déterminer le score mRS.",
            example="Le patient présente une faiblesse du côté gauche et nécessite une aide pour la marche, ce qui correspond à un score mRS de 3."
        )
        mRS: Annotated[int, Field(description="Échelle de Rankin Modifiée (0-6)", ge=0, le=6, example=3)]
        # visit_date: str = Field(..., description="Date au format jj/mm/aaaa")

    return GuidedDecodingParams(json=PatientInfoSchema.model_json_schema(), backend="lm-format-enforcer")


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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import time
# import re
# import pandas as pd
import torch
import transformers
from transformers import BitsAndBytesConfig
from datasets import Dataset
from functools import partial
from triton.testing import do_bench


DATASET_PATH = "data/dataset.csv"
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_ID = "epfl-llm/meditron-7b"  # <- not french + not instruct!
# MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MODEL_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
# MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
RUNTIME_QUANTIZATION = False
TORCH_DTYPE = torch.float16 if "hugging-quants" in MODEL_ID else torch.bfloat16


def main():
    """ Prompt a large language model with with medical questions
    """
    # Initialize model arguments
    model_kwargs = {"torch_dtype": TORCH_DTYPE}
    if RUNTIME_QUANTIZATION:
        assert not any(
            [s in MODEL_ID.lower() for s in ["-4bit", "-int4", "-quant"]]
        ), "Model is pre-quantized, you should not use runtime quantization!"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    # Build pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs=model_kwargs,
        device_map="auto",
    )
    
    # Prompt the model with all samples of the dataset
    dataset = Dataset.from_csv(DATASET_PATH)
    dataset = dataset.rename_columns({"Texte": "input_text", "mRS": "label"})
    process_fn = partial(process_sample, pipeline=pipeline)
    dataset = dataset.map(process_fn, batched=False)
    
    # Write the input and results to a csv file
    output_path = os.path.join("results", "%s.csv" % MODEL_ID)
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)


def process_sample(sample, pipeline):
    """ Take in a sample, format input text, and extract reasoning
    """
    messages = build_prompt(sample["input_text"])
    llm_outputs = pipeline(messages, max_new_tokens=1024)
    output_text = llm_outputs[0]["generated_text"][-1]["content"]
    question_output = post_process_llm_output(output_text)
    
    sample["reasoning"] = question_output["reasoning"]
    sample["prediction"] = question_output["answer"]
    
    return sample


def build_prompt(input_text: str) -> list[dict[str, str]]:
    """ Build a message list for prompting a large language model for the task at hand
    
    Args:
        input_text (str): core input text for the current sample

    Returns:
        list[dict[str, str]]: messages to prompt a large language model
    """
    system_prompt = """
Tu es un neuro-chirurgien et ta tâche est d’identifier le score sur l’Echelle de Rankin Modifiée Rankin (mRS) du patient, selon la lettre de sortie rédigée par un autre médecin.
L'Échelle de Rankin Modifiée (mRS) est utilisée pour mesurer le degré d'incapacité chez les patients ayant subi un accident vasculaire cérébral (AVC), comme suit:
0: Aucun symptôme
1: Aucune incapacité significative malgré des symptômes ; capable d'effectuer toutes les tâches et activités habituelles
2: Légère incapacité ; incapable d'effectuer toutes les activités antérieures, mais capable de s'occuper de ses propres affaires sans assistance
3: Incapacité modérée ; nécessitant une certaine aide, mais capable de marcher sans assistance
4: Incapacité modérément sévère ; incapable de marcher sans assistance et incapable de s'occuper de ses besoins corporels sans assistance
5: Incapacité sévère ; alité, incontinent et nécessitant des soins infirmiers constants et une attention continue
6: Décédé
    """
    intro_text = "Voici la lettre du patient:"
    output_specification = "Pour rédiger ta réponse, explique toutes les étapes de ton raisonnement, puis, à la toute fin de ta réponse, dans une nouvelle ligne, indique le mRS que tu prédis pour le patient en utilisant le format suivant: ```mRS: <nombre>```"
    user_prompt = "%s\n%s\n%s" % (intro_text, input_text, output_specification)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages


def post_process_llm_output(raw_output_text: str) -> dict[str, str]:
    """ Extract reasoning and answer from a raw text large language model output
    
    Args:
        raw_output_text (str): raw output from the large language model

    Returns:
        dict[str, str]: structured output from the large language model
    """
    lines = raw_output_text.split("\n")    
    reasoning = "\n".join(lines[:-1])
    answer = lines[-1].strip()
    
    return {"reasoning": reasoning, "answer": answer}


# def extract_label(model_output):
#     pattern = re.compile(r"(\d){1}")
#     return 


# def main_with_resource_monitoring():
#     # Clear cache and synchronize the GPU before forward pass
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
    
#     # Check GPU memory usage before forward pass
#     before_forward_memory = torch.cuda.memory_allocated()
    
#     # Perform function pass
#     main()
    
#     # Synchronize GPU after forward pass
#     torch.cuda.synchronize()

#     # Check GPU memory usage after forward pass
#     after_forward_memory = torch.cuda.memory_allocated()

#     # Peak memory usage during forward pass
#     peak_memory = torch.cuda.max_memory_allocated()
    
#     # Print GPU memory usage metrics
#     print(f"GPU memory usage before forward pass: {before_forward_memory / 1024**2:.2f} MB")
#     print(f"GPU memory usage after forward pass: {after_forward_memory / 1024**2:.2f} MB")
#     print(f"Peak GPU memory usage during forward pass: {peak_memory / 1024**2:.2f} MB")
    
#     # Check time computation
#     time.sleep(5)
#     milliseconds = do_bench(main)
    
#     # Print time computation metric
#     print(f"Time in milliseconds to run forward pass: {milliseconds} ms")


if __name__ == "__main__":
    main()
    
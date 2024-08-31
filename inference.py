import os
import re
import gc
import time
import numpy as np
import torch
import transformers
from llama_cpp import Llama
from transformers import BitsAndBytesConfig
from datasets import Dataset
from functools import partial
from utils import do_bench_custom, record_metrics, print_gpu_info
from sklearn.metrics import confusion_matrix
from config import Config as cfg


def main():
    """ Run benchmarks on different generative large language models
    """
    print("Benchmarking started\n")
    for run in cfg.RUNS:
    
        print("Benchmarking %s\n" % run["model_id"])
        print_gpu_info()
    
        inference_generator = create_inference_generator(**run)
        benchmark_one_model(inference_generator)
        
        print("Benchmarked %s\n" % run["model_id"])
    
    print("Benchmarking finished!")


def benchmark_one_model(
    inference_generator: transformers.pipelines.base.Pipeline|Llama,
) -> dict:
    """ Prompt a large language model with medical questions and computes
        metrics about computation time, GPU memory usage, and error rate
    
    Args:
        model_id (str): reference string to load model from huggingface
        runtime_quantize (bool): if True, runtime quantization is used    
            
    Returns:
        dict: benchmark metrics including time and memory usage
    """
    # Prompt the model with all samples of the dataset
    dataset = Dataset.from_csv(cfg.DATASET_PATH)
    dataset = dataset.rename_columns({"Texte": "input_text", "mRS": "label"})
    process_fn = lambda sample: process_sample(sample, inference_generator)
    if cfg.DEBUG: dataset = Dataset.from_dict(dataset[:2])
    
    # Measure computation time and GPU memory usage
    bench_fn = partial(dataset.map, function=process_fn, batch_size=1)
    times, memories, outputs = do_bench_custom(
        benchmarked_fn=bench_fn,
        n_repeats=2 if cfg.DEBUG else 10,
        return_outputs=True,
    )
    
    # Compute performance metrics
    cm_fn = lambda d: confusion_matrix(d["label"], d["prediction"], labels=list(range(-1, 7)))
    error_fn = lambda d: np.mean(np.array(d["prediction"]) != np.array(d["label"]))
    distance_fn = lambda d: np.mean(np.abs(np.array(d["prediction"]) - np.array(d["label"])))
    cm = np.sum([cm_fn(o) for o in outputs], axis=0)
    errors = torch.tensor([error_fn(o) for o in outputs], dtype=torch.float32)
    distances = torch.tensor([distance_fn(o) for o in outputs], dtype=torch.float32)
    
    # Combine all outputs and add them to the input dataset
    for key in ["reasoning", "answer", "prediction"]:
        for i, output in enumerate(outputs):
            dataset = dataset.add_column(f"{key}_{i:03}", output[key])
            
    # Write raw results and metrics to csv files
    output_path = os.path.join(cfg.RESULT_DIR, "%s_raw.csv" % inference_generator.name)
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    # Plot evaluation metrics to a png file
    metrics = [
        {"name": "Time per Sample", "unit": "s", "max_y": 100.0, "values": times / len(dataset)},
        {"name": "Peak VRAM Usage", "unit": "GB", "max_y": 100.0, "values": memories},
        {"name": "Error Rate", "unit": "%", "max_y": 1.0, "values": errors},
        {"name": "Distance", "unit": "mRS", "max_y": 1.0, "values": distances},
    ]
    result_path = output_path.replace("_raw.csv", "_metrics")
    record_metrics(result_path, confusion_matrix=cm, metrics=metrics)
    
    # Delete pipeline and free GPU memory cache
    del inference_generator
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def create_inference_generator(
    model_id: str,
    quantize_mode: str|None=None,
) -> transformers.pipelines.base.Pipeline|Llama:
    """ Create an LLM-based inference generator for solving a task
    
    Args:
        model_id (str): reference string to load model from huggingface
        quantize_mode (str, optional): define quantization used by the model
        
    Returns:
        transformers.pipelines.base.Pipeline|Llama: inference generator
    """
    # Initialize Llama-cpp-python model
    if ".gguf" in quantize_mode:
        inference_generator = Llama.from_pretrained(
            repo_id=model_id,
            filename=quantize_mode,
            n_gpu_layers=-1,
            n_ctx=cfg.MAX_INPUT_LENGTH,
            verbose=False,
        )
    
    # Initialize a classic transformer pipeline
    else:
        # Initialize model arguments
        torch_dtype = torch.bfloat16 if model_id not in cfg.AUTO_GPTQ_MODELS else torch.float16
        model_kwargs = {"torch_dtype": torch_dtype}
        if quantize_mode == "bits_and_bytes":
            assert not any(
                [s in model_id.lower() for s in ["-4bit", "-int4", "-quant"]]
            ), "Model is pre-quantized, you should not use runtime quantization!"
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        
        # Build pipeline
        inference_generator = transformers.pipeline(
            task="text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map="auto",
        )
        if model_id in cfg.AUTO_GPTQ_MODELS:
            from auto_gptq import exllama_set_max_input_length
            inference_generator.model = \
                exllama_set_max_input_length(
                    inference_generator.model,
                    max_input_length=cfg.MAX_INPUT_LENGTH,
                )
    
    # Return inference generator after giving it a name
    inference_generator.name = "%s_quantized_%s" % (model_id, quantize_mode)
    return inference_generator


def process_sample(
    sample: dict[str, str],
    inference_generator: transformers.pipelines.base.Pipeline|Llama,
) -> dict[str, str]:
    """ Process a sample by formatting the input text, prompting an LLM, and
        extracting reasoning and predictions.

    Args:
        sample (dict[str, str]): sample including input text
        inference_generator (transformers.pipelines.base.Pipeline, Llama):
            pipeline for language model inference

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    messages = build_prompt(sample["input_text"])
    
    # Case where a classic pipeline is used
    if isinstance(inference_generator, transformers.pipelines.base.Pipeline):
        llm_outputs = inference_generator(
            messages,
            max_new_tokens=1024,
            pad_token_id=inference_generator.tokenizer.eos_token_id,  # would be set in any case
        )
        output_text = llm_outputs[0]["generated_text"][-1]["content"]
    
    # Case where a llama-cpp-python model is used
    if isinstance(inference_generator, Llama):
        outputs = inference_generator.create_chat_completion(messages)
        output_text = outputs["choices"][0]["message"]["content"]
    
    question_output = extract_reasoning_and_prediction(output_text)
    sample.update(question_output)
    
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
L'Échelle de Rankin Modifiée (mRS) est utilisée pour mesurer le degré d'incapacité chez les patients ayant subi un accident vasculaire cérébral (AVC), comme suit :
0 : Aucun symptôme
1 : Aucune incapacité significative malgré des symptômes ; capable d'effectuer toutes les tâches et activités habituelles
2 : Légère incapacité ; incapable d'effectuer toutes les activités antérieures, mais capable de s'occuper de ses propres affaires sans assistance
3 : Incapacité modérée ; nécessitant une certaine aide, mais capable de marcher sans assistance
4 : Incapacité modérément sévère ; incapable de marcher sans assistance et incapable de s'occuper de ses besoins corporels sans assistance
5 : Incapacité sévère ; alité, incontinent et nécessitant des soins infirmiers constants et une attention continue
6 : Décédé
    """
    intro_text = "Voici la lettre du patient :"
    output_specification = "Pour rédiger ta réponse, explique toutes les étapes de ton raisonnement, puis, à la toute fin de ta réponse, dans une nouvelle ligne, indique le mRS que tu prédis pour le patient en utilisant le format suivant : ```mRS : <nombre>```"
    user_prompt = "\n".join((intro_text, input_text, output_specification))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages


def extract_reasoning_and_prediction(
    llm_raw_output_text: str,
) -> dict[str, str]:
    """ Extract reasoning and answer from the raw output of an LLM
    
    Args:
        raw_output_text (str): raw output from the LLM

    Returns:
        dict[str, str]: structured output from the LLM
    """
    # Extract reasoning and text answer from raw output 
    lines = llm_raw_output_text.split("\n")    
    reasoning = "\n".join(lines[:-1])
    answer = lines[-1].strip()
    
    # Extract prediction from text answer
    pattern = re.compile(r'(?:mrs[:\s]*|\b)([0-6])\b', re.IGNORECASE)
    match = pattern.search(answer)
    prediction = int(match.group(1)) if match else -1  # -1 being "bad answer"
    
    return {"reasoning": reasoning, "answer": answer, "prediction": prediction}


if __name__ == "__main__":
    main()
    
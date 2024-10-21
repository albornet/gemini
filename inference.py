import os
import re
import gc
import argparse
import itertools
import numpy as np
import torch
import torch.multiprocessing as torch_mp
from llama_cpp import Llama
from datasets import Dataset
from functools import partial
from utils import do_bench_custom, record_metrics, print_gpu_info
from sklearn.metrics import confusion_matrix
from config import Config as cfg

parser = argparse.ArgumentParser(description="Benchmark LLMs on healthcare tasks")
parser.add_argument("-t", "--runtype", default="debug", help="Type of run")
args = parser.parse_args()
RUNTYPE = args.runtype  # very_small, small, big, all


def main():
    """ Run benchmarks on different generative large language models in separate
        processes to avoid GPU memory leak or accumulation
    """
    # Select runs based on script argument
    if RUNTYPE == "all":
        run_args_list = list(itertools.chain(*cfg.RUN_DICT.values()))
    else:
        assert RUNTYPE in cfg.RUN_DICT, "Invalid runtype argument"
        run_args_list = cfg.RUN_DICT[RUNTYPE]
    
    # Benchmark all models sequentially, spawning one process per benchmark
    print("Benchmarking started, using %s model types\n" % RUNTYPE)
    for run_args in run_args_list:
        process = torch_mp.Process(target=clean_benchmark_run, args=(run_args,))
        process.start()  # spawn a new process for each benchmark run
        process.join()   # wait for the process to complete before continuing
        
    # Success message
    print("Benchmarking finished!")


def clean_benchmark_run(run_args: dict[str, str]) -> None:
    """ Runs the benchmark for a single model in a separate process
    """
    # Check gpu info and initialize inference generator
    print_gpu_info()
    inference_generator = create_inference_generator(**run_args)
    
    # Actual benchmark running part
    try:
        print("Benchmarking %s\n" % inference_generator.name)
        benchmark_one_model(inference_generator)
        print("Benchmarked %s\n" % inference_generator.name)
    
    # Data cleaning part to free GPU memory
    finally:
        if "inference_generator" in locals(): del inference_generator
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        
def benchmark_one_model(
    inference_generator: Llama,
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
    process_fn = lambda sample: process_sample(sample, inference_generator)
    if cfg.DEBUG: dataset = Dataset.from_dict(dataset[:5])
    
    # Measure computation time and GPU memory usage
    bench_fn = partial(dataset.map, function=process_fn, batch_size=1, load_from_cache_file=False)
    times, memories, outputs = do_bench_custom(
        benchmarked_fn=bench_fn,
        n_repeats=2 if cfg.DEBUG else cfg.N_INFERENCE_REPEATS,
        return_outputs=True,
    )
    times = times / len(dataset)  # since we want time per sample
    
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
    output_path = os.path.join(cfg.RESULT_DIR, "%s_raw.csv" % inference_generator.path)
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    # Plot evaluation metrics to a png file
    metrics = [
        {"name": "Time per Sample", "unit": "s", "max_y": 100.0, "values": times},
        {"name": "Peak VRAM Usage", "unit": "GB", "max_y": 60.0, "values": memories},
        {"name": "Error Rate", "unit": "%", "max_y": 1.0, "values": errors},
        {"name": "Distance", "unit": "mRS", "max_y": 1.0, "values": distances},
    ]
    result_path = output_path.replace("_raw.csv", "_metrics")
    record_metrics(result_path, confusion_matrix=cm, metrics=metrics)


def create_inference_generator(
    repo_name: str,
    model_id: str,
    quantize_mode: str,
) -> Llama:
    """ Create an LLM-based inference generator for solving a task
    
    Args:
        model_id (str): reference string to load model from huggingface
        quantize_mode (str, optional): define quantization used by the model
        
    Returns:
        Llama: inference generator
    """
    # Initialize Llama-cpp-python model
    repo_id = "/".join((repo_name, model_id))  # always "/" for huggingface or Llama
    inference_generator = Llama.from_pretrained(
        repo_id=repo_id,
        filename=quantize_mode,
        n_gpu_layers=-1,
        n_ctx=cfg.MAX_CONTEXT_LENGTH,
        flash_attn=cfg.USE_FLASH_ATTENTION,
        verbose=False,
    )
    
    # Return inference generator after giving it a name
    inference_generator.path = os.path.join(repo_name, model_id, quantize_mode)
    inference_generator.name = "%s_quantized_%s" % (repo_id, quantize_mode)
    return inference_generator


def process_sample(
    sample: dict[str, str],
    inference_generator: Llama,
) -> dict[str, str]:
    """ Process a sample by formatting the input text, prompting an LLM, and
        extracting reasoning and predictions.

    Args:
        sample (dict[str, str]): sample including input text
        inference_generator (Llama): pipeline for language model inference

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    # Prompt model and collect output
    messages = build_prompt(sample["input_text"])
    outputs = inference_generator.create_chat_completion(messages, logprobs=1)
    
    # Extract text response, reasoning, and model prediction
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
Tu es un neuro-chirurgien et ta tâche est d’identifier le score sur l’Echelle de Rankin Modifiée Rankin (mRS) du patient en fonction de son état après sa sortie, tel qu'il est décrit dans la lettre de sortie rédigée par un autre médecin.
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
    torch_mp.set_start_method("spawn", force=True)
    main()
    
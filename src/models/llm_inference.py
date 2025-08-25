from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llama_cpp import Llama
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Any
from functools import partial

from src.data.output_guiding import (
    create_pydantic_model_from_schema_dict,
    create_output_guide,
    extract_structured_output,
)


def _infer_vllm(
    model: LLM,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float=1.0,
    top_p: float=1.0,
    output_guide: dict[str, Any]|None=None,
):
    """
    Helper function to run inference using vLLM.
    """
    # Build sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        guided_decoding=output_guide,
    )

    # Run vLLM model on all dataset's prompts
    outputs = model.generate(dataset["prompt"], sampling_params=sampling_params)
    output_texts = [output.outputs[0].text.strip() for output in outputs]

    return output_texts


def _infer_llama_cpp(
    model: Llama,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float=1.0,
    top_p: float=1.0,
    output_guide: dict[str, Any]|None=None,
):
    """
    Helper function to run inference using llama-cpp
    """
    # Build response format
    output_texts = []
    if output_guide is not None:
        response_format = {"type": "json_object", "schema": output_guide}
    else:
        response_format = None

    # Run llama-cpp model on the dataset
    for messages in tqdm(dataset["messages"], desc="Generating inferences (llama-cpp)"):
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
        )
        output_texts.append(response["choices"][0]["message"]["content"].strip())

    return output_texts


def _infer_huggingface(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any]|None=None,
):
    """
    Helper function to run inference using HuggingFace
    """
    # TODO: SEE IF WE CAN USE AN OUTPUT GUIDE WITH NATIVE HUGGINGFACE BACKEND
    output_texts = []
    for prompt in tqdm(dataset["prompt"], desc="Generating inferences (HF)"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
        generated_tokens = output[0, inputs["input_ids"].shape[-1]:]
        output_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    
    return output_texts


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM | Llama | LLM,
    tokenizer: AutoTokenizer,
    inference_backend: str,
    use_output_guide: bool = False,
    output_schema: dict[str, Any] | None = None,
    output_schema_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    *args, **kwargs,
) -> dict[str, str]:
    """
    Run inference on dataset samples using vLLM, llama-cpp, or HuggingFace backends.
    """
    # Build output guide if requested
    output_guide = create_output_guide(
        inference_backend=inference_backend,
        output_schema_dict=output_schema,
        output_schema_name=output_schema_name,
    ) if use_output_guide else None

    # Run inference
    match inference_backend:
        case "vllm":
            output_texts = _infer_vllm(model, dataset, max_new_tokens, temperature, top_p, output_guide)
        case "llama-cpp":
            output_texts = _infer_llama_cpp(model, dataset, max_new_tokens, temperature, top_p, output_guide)
        case "huggingface":
            output_texts = _infer_huggingface(model, tokenizer, dataset, max_new_tokens, temperature, top_p)
        case _:
            raise ValueError(f"Unknown inference backend: {inference_backend}")

    # Postprocess with schema
    output_schema_model = create_pydantic_model_from_schema_dict(
        schema_dict=output_schema,
        model_name=output_schema_name,
    )
    dataset = dataset.add_column("output_text", output_texts)
    dataset = dataset.map(
        function=partial(extract_structured_output, output_schema_model=output_schema_model),
        desc="Extracting model predictions",
    )

    return dataset

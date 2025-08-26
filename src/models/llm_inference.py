from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llama_cpp import Llama
from vllm import LLM, SamplingParams, RequestOutput
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
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Helper function to run inference using vLLM.
    Generates `n_inference_repeats` outputs for each prompt.
    """
    # Build sampling parameters, using 'n' for repeated inferences per prompt.
    sampling_params = SamplingParams(
        n=n_inference_repeats,  # Generate N completions for each prompt
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        guided_decoding=output_guide,
    )

    # Run vLLM model on all dataset's prompts (single, efficient batch call)
    outputs: list[RequestOutput] = model.generate(dataset["prompt"], sampling_params=sampling_params)
    
    # Each 'request_output' in 'outputs' corresponds to a prompt and contains 'n' completions.
    output_texts = [
        [completion.text.strip() for completion in request_output.outputs]
        for request_output in outputs
    ]

    return output_texts


def _infer_llama_cpp(
    model: Llama,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Helper function to run inference using llama-cpp.
    Generates `n_inference_repeats` outputs for each prompt.
    """
    # Build response format
    if output_guide is not None:
        response_format = {"type": "json_object", "schema": output_guide}
    else:
        response_format = None

    # Run llama-cpp model on the dataset
    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Generating inferences (llama-cpp)"):
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n_inference_repeats,  # Generate N choices for the prompt
            response_format=response_format,
        )
        all_outputs.append([
            choice["message"]["content"].strip() for choice in response["choices"]
        ])

    return all_outputs


def _infer_huggingface(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Helper function to run inference using HuggingFace.
    Generates `n_inference_repeats` outputs for each prompt.
    """
    # TODO: SEE IF WE CAN USE AN OUTPUT GUIDE WITH NATIVE HUGGINGFACE BACKEND
    if output_guide is not None:
        print("Warning: HuggingFace native inference does not support guided output in this setup.")

    # To generate different sequences, sampling must be enabled
    do_sample = True if n_inference_repeats > 1 and temperature > 0.0 else False
    all_outputs = []
    for prompt in tqdm(dataset["prompt"], desc="Generating inferences (HF)"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Use num_return_sequences to generate multiple outputs efficiently
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_inference_repeats, # Generate N sequences
            do_sample=do_sample,
        )

        # The output tensor now has a batch size of n_inference_repeats
        generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
        this_prompt_outputs = [
            tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_tokens
        ]
        all_outputs.append(this_prompt_outputs)

    return all_outputs


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM | Llama | LLM,
    tokenizer: AutoTokenizer,
    inference_backend: str,
    n_inference_repeats: int,
    use_output_guide: bool = False,
    output_schema: dict[str, Any] | None = None,
    output_schema_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    *args, **kwargs,
) -> Dataset:
    """
    Run inference on dataset samples using vLLM, llama-cpp, or HuggingFace backends.
    """
    # Build output guide if requested
    output_guide = create_output_guide(
        inference_backend=inference_backend,
        output_schema_dict=output_schema,
        output_schema_name=output_schema_name,
    ) if use_output_guide else None

    # Select inference backend
    match inference_backend:
        case "vllm": inference_fn = _infer_vllm
        case "llama-cpp": inference_fn = _infer_llama_cpp
        case "huggingface": inference_fn = _infer_huggingface
        case _: raise ValueError(f"Unknown inference backend: {inference_backend}")

    # Run inference
    output_texts = inference_fn(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        n_inference_repeats=n_inference_repeats,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        output_guide=output_guide,
    )  # -> this is a list of n_sample lists of shape n_inference_repeats

    # Add raw output text and structured outputs to the dataset
    output_schema_model = create_pydantic_model_from_schema_dict(
        schema_dict=output_schema,
        model_name=output_schema_name,
    )
    transposed_output_texts = list(zip(*output_texts))  # per-model list
    for i, model_outputs in enumerate(transposed_output_texts):

        # Add raw output text, adding the model index
        column_name = f"output_text_{i:03d}"
        dataset = dataset.add_column(name=column_name, column=model_outputs)

        # Add structured output, keeping the model index in the output column
        def mapping_wrapper(sample):
            structured_dict = extract_structured_output(
                sample=sample,
                output_schema_model=output_schema_model,
                col_to_structure=column_name,
            )
            return {f"{k}_{i:03d}": v for k, v in structured_dict.items()}
        
        dataset = dataset.map(mapping_wrapper, desc="Extracting model predictions")

    return dataset

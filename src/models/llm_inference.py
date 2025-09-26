import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import AutoModelForCausalLM
from datasets import Dataset
from llama_cpp import Llama
from vllm import LLM, SamplingParams, RequestOutput
from openai import OpenAI, AsyncOpenAI, InternalServerError, APIConnectionError
from tqdm import tqdm
from tqdm.asyncio import tqdm as anext
from typing import Any
from functools import partial
from vllm.sampling_params import GuidedDecodingParams
from src.data.prompting import build_prompt
from src.data.output_guiding import (
    create_pydantic_model_from_schema_dict,
    extract_structured_output,
)


def _infer_vllm(
    model: LLM,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    json_schema: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference with vLLM directly (faster than querying vLLM-serve)
    """
    # Build sampling parameters
    sampling_params = SamplingParams(
        n=n_inference_repeats,  # several completions per prompt
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        guided_decoding=GuidedDecodingParams(json=json_schema),
    )

    # Build prompts using model tokenizer by mapping dataset messages
    tokenizer_fn = partial(
        build_prompt,
        tokenizer=model.get_tokenizer(),
        add_generation_prompt=True,
        enable_thinking=False,
    )
    dataset = dataset.map(tokenizer_fn, desc="Building prompts for vLLM")

    # Run vLLM model on all dataset's prompts (single, efficient batch call)
    outputs: list[RequestOutput] = model.generate(dataset["prompt"], sampling_params=sampling_params)
    
    # Each request_output in outputs corresponds to a prompt and contains n completions
    output_texts = [
        [completion.text.strip() for completion in request_output.outputs]
        for request_output in outputs
    ]

    return output_texts


def _extract_outputs_vllm(choices: list) -> list[str]:
    """
    Extract content from chat completion choices, falling back to reasoning content
    """
    outputs = []
    for choice in choices:
        content = choice.message.content
        if content is None:
            content = getattr(choice.message, "reasoning_content", None)
        outputs.append((content or "").strip())
        
    return outputs


def _infer_vllm_serve(
    model: OpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference by querying the vLLM server using the chat completion API
    """
    client = model
    model_name = client.models.list().data[0].id

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((InternalServerError, ConnectionError)),
        reraise=True,
    )
    def generate_vllm_outputs(messages: list[dict[str, str]]):
        """Single-shot API call function"""
        extra_body = {"guided_json": output_guide} if output_guide else None
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n_inference_repeats,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        return _extract_outputs_vllm(chat_completion.choices)

    # Query the server for each task separately
    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Querying vLLM server"):
        all_outputs.append(generate_vllm_outputs(messages))
        time.sleep(1)  # avoid overwhelming the server

    return all_outputs


async def _infer_vllm_serve_async(
    model: AsyncOpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_guide: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference by querying the vLLM server asynchronously
    """
    client = model
    model_name = (await client.models.list()).data[0].id

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((InternalServerError, ConnectionError)),
        reraise=True,
    )
    async def generate_vllm_outputs(messages: list[dict[str, str]]):
        """Single-shot async API call function"""
        extra_body = {"guided_json": output_guide} if output_guide else None
        chat_completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n_inference_repeats,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        return _extract_outputs_vllm(chat_completion.choices)

    # Query the server for all tasks concurrently
    tasks = [generate_vllm_outputs(messages) for messages in dataset["messages"]]
    all_outputs = await anext.gather(*tasks, desc="Querying vLLM server (async)")

    return all_outputs


def _infer_llama_cpp(
    model: Llama,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    json_schema: dict[str, Any] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference with llama-cpp-python
    """
    # Build response format
    response_format = None
    if json_schema is not None:
        response_format = {"type": "json_object", "schema": json_schema}

    # Run llama-cpp model on the dataset
    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Generating inferences (llama-cpp)"):
        prompt_outputs = []

        # Loop n_inference_repeats times for each message
        for _ in range(n_inference_repeats):
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )

            # Each response has one choice, so we get it at index 0
            content = response["choices"][0]["message"]["content"]
            prompt_outputs.append(content.strip())
            
        all_outputs.append(prompt_outputs)

    return all_outputs


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM | Llama | OpenAI,
    inference_backend: str,
    n_inference_repeats: int,
    use_output_guide: bool = False,
    output_schema_dict: dict[str, Any] | None = None,
    output_schema_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    *args, **kwargs,
) -> Dataset:
    """
    Run inference on dataset samples using vLLM, llama-cpp, or HuggingFace backends.
    """
    # # Remove any reasoning field from output schema for an already reasoning model
    # THIS IS NOT IMPLEMENTED YET HERE BECAUSE I DID NOT FIGURE OUT WHETHER OR NOT
    # QWEN3-LIKE MODELS CAN BE RUN BOTH WITH STRUCTURED OUTPUT GUIDING AND ENABLE
    # THINKING IN VLLM!!
    # is_reasoning = ("enable_thinking" in model.get_tokenizer().chat_template)
    # if is_reasoning:
    #     if "reasoning" in output_schema_dict:
    #         del output_schema_dict["reasoning"]
    #         print("Removed 'reasoning' field from output schema for reasoning model.")

    # Build output guide if requested (i.e., output schema influences LLM's inference)
    if not use_output_guide:
        output_guide = None
    else:
        output_guide = create_pydantic_model_from_schema_dict(
            schema_dict=output_schema_dict,
            model_name=output_schema_name,
        ).model_json_schema()

    # Define arguments for model inference 
    infer_args = {
        "model": model,
        "dataset": dataset,
        "n_inference_repeats": n_inference_repeats,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "output_guide": output_guide,
        **kwargs,
    }

    # Run inference using the correct backend
    match inference_backend:
        case "vllm": output_texts = _infer_vllm(**infer_args)
        case "vllm-serve": output_texts = _infer_vllm_serve(**infer_args)
        case "llama-cpp": output_texts = _infer_llama_cpp(**infer_args)
        case "vllm-serve-async": output_texts = asyncio.run(_infer_vllm_serve_async(**infer_args))
        case _: raise ValueError(f"Unknown inference backend: {inference_backend}")

    # Extract Pydantic style output schema from the configuration
    transposed_output_texts = list(zip(*output_texts))
    for i, model_outputs in enumerate(transposed_output_texts):

        # Add raw output text, adding the model index
        column_name = f"output_text_{i:03d}"
        dataset = dataset.add_column(name=column_name, column=model_outputs)

        # Define function to add structured output, keeping the model index in the output column
        mapping_fn = partial(
            _map_and_structure_output,
            output_schema_dict=output_schema_dict,
            output_schema_name=output_schema_name,
            col_to_structure=column_name,
            inference_idx=i,
        )
        dataset = dataset.map(mapping_fn, desc="Extracting model predictions")

    print("All LLM outputs were parsed.")
    return dataset


def _map_and_structure_output(
    sample: dict[str, Any],
    output_schema_dict: dict[str, Any],
    output_schema_name: str,
    col_to_structure: str,
    inference_idx: int,
) -> dict[str, Any]:
    """
    Extracts structured data from a single sample's text output and add the index
    of the inference that generated that output
    """
    # Recreate the model from the serializable inputs
    output_schema_model = create_pydantic_model_from_schema_dict(
        schema_dict=output_schema_dict,
        model_name=output_schema_name,
    )

    structured_dict = extract_structured_output(
        sample=sample,
        output_schema_model=output_schema_model,
        col_to_structure=col_to_structure,
    )

    # Add the inference index to make column names unique
    return {f"{k}_{inference_idx:03d}": v for k, v in structured_dict.items()}

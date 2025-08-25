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


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM|Llama|LLM,
    tokenizer: AutoTokenizer,
    inference_backend: str,
    use_output_guide: bool=False,
    output_schema: dict[str, Any]|None=None,
    output_schema_name: str|None=None,
    max_generated_tokens: int=512,
    temperature: float=1.0,
    top_p: float=1.0,
    *args, **kwargs,
) -> dict[str, str]:
    """
    Process a sample by formatting the input text, prompting an LLM,
    and extracting reasoning and predictions.

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    # Get output guide if required
    if use_output_guide:
        output_guide = create_output_guide(
            inference_backend=inference_backend,
            output_schema_dict=output_schema,
            output_schema_name=output_schema_name,
        )
    else:
        output_guide = None

    # Use LLM inference to process the dataset
    output_texts = []
    match inference_backend:

        case "vllm":
            sampling_params = SamplingParams(
                max_tokens=max_generated_tokens,
                temperature=temperature,
                top_p=top_p,
                guided_decoding=output_guide,
            )
            outputs = model.chat(dataset["messages"], sampling_params=sampling_params)
            output_texts = [output.outputs[0].text.strip() for output in outputs]
            
        case "llama-cpp":
            if output_guide:
                response_format = {"type": "json_object", "schema": output_guide}
            else:
                response_format = None
            for messages in tqdm(dataset["messages"], desc="Generating inferences"):
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_generated_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                )
                output_text = response["choices"][0]["message"]["content"].strip()
                output_texts.append(output_text)

        case "huggingface":
            # TODO: FIGURE OUT HOW I CAN INCLUDE OUTPUT GUIDING HERE, OR DROP HF?
            for prompt in tqdm(dataset["prompt"], desc="Generating inferences"):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_generated_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temp=temperature,
                    top_p=top_p,
                )
                generated_tokens = output[0, inputs["input_ids"].shape[-1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                output_texts.append(output_text)
    
    # Update the dataset with the output the LLM
    output_schema_model = create_pydantic_model_from_schema_dict(
        schema_dict=output_schema,
        model_name=output_schema_name,
    )
    dataset = dataset.add_column("output_text", output_texts)
    dataset = dataset.map(
        function=partial(
            extract_structured_output,
            output_schema_model=output_schema_model,
        ),
        desc="Extracting model predictions",
    )

    return dataset

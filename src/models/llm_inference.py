import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llama_cpp import Llama
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Any
from pydantic import ValidationError
from pydantic_core import PydanticUndefinedType
from src.utils.run_utils import load_config
from src.data.output_guiding import create_pydantic_model_from_schema_dict
from src.data.output_guiding import create_output_guide

cfg = load_config()
OUTPUT_SCHEMA_MODEL = create_pydantic_model_from_schema_dict(
    schema_dict=cfg["output_schema"],
    model_name=cfg["output_schema_name"],
)


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM|Llama|LLM,
    tokenizer: AutoTokenizer,
) -> dict[str, str]:
    """ Process a sample by formatting the input text, prompting an LLM, and
        extracting reasoning and predictions.

    Args:
        TODO: CORRECT THIS !!! sample (dict[str, str]): sample including input text
        model (AutoModelForCausalLM): language model used for inference
        tokenizer (AutoTokenizer): tokenizer used by the inference model

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    # Get output guide if required
    if cfg["USE_OUTPUT_GUIDE"]:
        output_guide = create_output_guide(inference_backend=cfg["INFERENCE_BACKEND"])
    else:
        output_guide = None

    # Use LLM inference to process the dataset
    output_texts = []
    match cfg["INFERENCE_BACKEND"]:

        case "vllm":
            sampling_params = SamplingParams(
                max_tokens=cfg["MAX_GENERATED_TOKENS"],
                temperature=cfg["TEMPERATURE"],
                top_p=cfg["TOP_P"],
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
                    max_tokens=cfg["MAX_GENERATED_TOKENS"],
                    temperature=cfg["TEMPERATURE"],
                    top_p=cfg["TOP_P"],
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
                    max_new_tokens=cfg["MAX_GENERATED_TOKENS"],
                    pad_token_id=tokenizer.eos_token_id,
                    temp=cfg["TEMPERATURE"],
                    top_p=cfg["TOP_P"],
                )
                generated_tokens = output[0, inputs["input_ids"].shape[-1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                output_texts.append(output_text)
    
    # Update the dataset with the output the LLM
    dataset = dataset.add_column("output_text", output_texts)
    dataset = dataset.map(extract_structured_output, desc="Extracting model predictions")

    return dataset


def extract_structured_output(sample: dict[str, Any]) -> dict[str, Any]:
    """ Extract structured output from raw model output vaidating with pydantic

        Args:
        - raw_output (str): raw output from the LLM
    
        Returns:
        - dict[str, Any]: structured and validated output from the LLM
    """
    # Extract model's raw answer
    raw_output = sample.get("output_text")
    if raw_output is None:
        raise KeyError("The sample is missing the 'output_text' key. Cannot process.")
    
    # Try direct JSON parse
    try:
        validated_output = OUTPUT_SCHEMA_MODEL.model_validate_json(raw_output.strip())
        return validated_output.model_dump()
    
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Pydantic validation/JSON parsing failed: {e}. Attempting lenient parsing.")

    # If direct validation fails, try extracting substring between first "{" and last "}"
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        json_candidate = raw_output[start:end]
        validated_output = OUTPUT_SCHEMA_MODEL.model_validate_json(json_candidate)
        return validated_output.model_dump()
    
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        print(f"Lenient parsing failed: {e}. Returning default values.")
    
    # Build default output if all parsing methods failed
    default_output = {}
    for field_name, field in OUTPUT_SCHEMA_MODEL.model_fields.items():
        if field.default_factory is None:
            if isinstance(field.default, PydanticUndefinedType):
                default_output[field_name] = None
            else:
                default_output[field_name] = field.default
        else:
            default_output[field_name] = field.default_factory()

    return default_output

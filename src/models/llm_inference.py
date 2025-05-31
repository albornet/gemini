import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llama_cpp import Llama
from vllm import LLM, SamplingParams
from tqdm import tqdm
from src.utils.run_utils import load_config
from src.data.prompt_utils import get_output_guide

cfg = load_config()


def process_samples(
    dataset: Dataset,
    model: AutoModelForCausalLM|Llama|LLM,
    tokenizer: AutoTokenizer,
) -> dict[str, str]:
    """ Process a sample by formatting the input text, prompting an LLM, and
        extracting reasoning and predictions.

    Args:
        sample (dict[str, str]): sample including input text
        model (AutoModelForCausalLM): language model used for inference
        tokenizer (AutoTokenizer): tokenizer used by the inference model

    Returns:
        dict[str, str]: updated sample with extracted reasoning and predictions
    """
    # Get output guide if required
    if cfg.USE_OUTPUT_GUIDE:
        output_guide = get_output_guide(
            inference_backend=cfg.INFERENCE_BACKEND,
            max_generated_tokens=cfg.MAX_GENERATED_TOKENS,
        )
    else:
        output_guide = None

    # Use LLM inference to process the dataset
    output_texts = []
    match cfg.INFERENCE_BACKEND:

        case "vllm":
            sampling_params = SamplingParams(
                max_tokens=cfg.MAX_GENERATED_TOKENS,
                temperature=cfg.TEMPERATURE,
                top_p=cfg.TOP_P,
                guided_decoding=output_guide,
            )
            outputs = model.chat(dataset["messages"], sampling_params=sampling_params)
            output_texts = [output.outputs[0].text.strip() for output in outputs]
            
        case "llama-cpp":
            for messages in tqdm(dataset["messages"], desc="Generating inferences"):
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=cfg.MAX_GENERATED_TOKENS,
                    temperature=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                )
                output_text = response["choices"][0]["message"]["content"].strip()
                output_texts.append(output_text)
                
        case "huggingface":
            for prompt in tqdm(dataset["prompt"], desc="Generating inferences"):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=cfg.MAX_GENERATED_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    temp=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                )
                generated_tokens = output[0, inputs["input_ids"].shape[-1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                output_texts.append(output_text)
    
    # Update the dataset with the output the LLM
    dataset = dataset.add_column("output_text", output_texts)
    dataset = dataset.map(
        lambda s: extract_reasoning_and_prediction(s["output_text"]),
        desc="Extracting model predictions",
    )

    return dataset


def extract_reasoning_and_prediction(raw_output: str) -> dict:
    """ Extract reasoning and mRS score from raw model output
    
        Args:
            raw_output_text (str): raw output from the LLM

        Returns:
            dict[str, str]: structured output from the LLM
    """
    # Try direct JSON parse
    try:
        formatted_output = json.loads(raw_output.strip())
        return normalize_output_keys(formatted_output)
    except json.JSONDecodeError:
        print("Parsing LLM output failed, falling back to more lenient parsing")
        pass

    # Try extracting substring between first "{" and last "}"
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        json_candidate = raw_output[start:end]
        formatted_output = json.loads(json_candidate)
        return normalize_output_keys(formatted_output)
    except (ValueError, json.JSONDecodeError):
        print("Lenient parsing failed as well, falling back to regex matching")
        pass  # continue to fallback 2

    # Fallback on regex-based strategies
    matches = list(re.finditer(r"mRS[\s:;-]{0,10}([0-6])\b", raw_output, re.IGNORECASE))
    if matches:
        mrs_score = int(matches[-1].group(1))  # last matching digit close to "mRS"
    else:
        all_scores = re.findall(r"\b[0-6]\b", raw_output)  # ast digit number
        mrs_score = int(all_scores[-1]) if all_scores else -1
    
    formatted_output = {"reasoning": raw_output.strip(), "prediction": mrs_score}
    return normalize_output_keys(formatted_output)


def normalize_output_keys(output_dict: dict) -> dict:
    """ Normalize formatted LLM output for mRS extraction
    """
    reasoning = ""  # indicating "no reasoning"
    prediction = -1  # indicating "mRS not found"
    for key, value in output_dict.items():
        if key.lower() == "reasoning":
            reasoning = value
        if key.lower() == "mrs":
            if -1 <= value <= 6:
                prediction = value

    return {"reasoning": reasoning, "prediction": prediction}

from typing import Union
from transformers import AutoTokenizer


def build_prompt(
    sample: dict[str, str],
    cfg: dict,
    tokenizer: AutoTokenizer=None,
) -> dict[str, Union[list, str]]:
    """ Takes in a data sample, builds Hugging Face style messages, and optionally
        tokenizes the associated prompt using the provided config

    Args:
        sample: dictionary containing the input text, expected as sample["input_text"]
        cfg: configuration dictionary containing prompt templates and context data
        tokenizer (optional): instance applying the chat template, if required

    Returns:
        A dictionary containing:
            - "messages" (list): list of messages for the LLM.
            - "prompt" (str): tokenized prompt string if a tokenizer is provided, otherwise None.
    """
    # Get prompt info from configuration
    system_template: str = cfg["prompt_templates"]["system_template"]
    user_template: str = cfg["prompt_templates"]["user_template"]
    context_data: dict[str, str] = cfg["context_data"]

    # Populate system and user prompt templates
    system_prompt_content = system_template.format(**context_data)
    user_prompt_content = user_template.format(input_text=sample["input_text"])
    
    # Create huggingface-formatted message stream
    messages = [
        {"role": "system", "content": system_prompt_content.strip()},
        {"role": "user", "content": user_prompt_content.strip()},
    ]

    # Apply the chat template if a tokenizer is provided
    prompt_str = None
    if tokenizer is not None:
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return {"messages": messages, "prompt": prompt_str}


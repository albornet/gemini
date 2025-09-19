import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    """
    Loads the training configuration from a YAML file
    """
    # TODO: CHANGE THIS FOR USING RUN_UTILS!!
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config_path: str):
    """
    Main function to run the fine-tuning process
    """
    cfg = load_config(config_path)

    # Load Dataset and Apply prompt template
    print("Loading and formatting dataset...")
    dataset = load_dataset("csv", data_files=cfg['data']['dataset_path'], split="train")
    
    # The global tokenizer is defined later, but needed for the mapping function
    global tokenizer  # TODO: CHANGE THIS FOR USING RUN_UTILS!!!
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_path'])
    
    # The SFTTrainer expects a single text column
    formatted_dataset = dataset.map(
        lambda sample: create_prompt(sample, cfg),
        remove_columns=list(dataset.features) # Remove old columns
    )
    print("Dataset formatted successfully.")
    print(f"\nExample of a formatted prompt:\n{'-'*50}\n{formatted_dataset[0]['formatted_prompt']}\n{'-'*50}\n")

    # Configure QLoRA for efficient fine-tuning
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    peft_config = LoraConfig(**cfg['lora_config'])

    # Load the model to finetune and corresponding
    print(f"Loading base model: {cfg['model']['base_model_path']}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['base_model_path'],
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=cfg['model']['new_model_path'],
        **cfg['training_args']
    )

    # Initialize the SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="formatted_prompt", # The column with our formatted text
        max_seq_length=1024, # Adjust based on your VRAM and data
        tokenizer=tokenizer,
        args=training_args,
        packing=False, # Set to True for speed-up on long sequences, if needed
    )

    # Run model training
    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {cfg['model']['new_model_path']}...")
    trainer.save_model()
    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM for mRS classification.")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
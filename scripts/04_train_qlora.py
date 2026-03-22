#!/usr/bin/env python3
"""QLoRA fine-tuning script for RunPod A100. DO NOT RUN LOCALLY — requires GPU."""

import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Model selection with fallback
PRIMARY_MODEL = "nvidia/Nemotron-Mini-4B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

TRAIN_PATH = "data/train.json"
VAL_PATH = "data/val.json"
OUTPUT_DIR = "./cricketmind-lora"


def format_example(example):
    """Format training example into chat template."""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def load_dataset_from_json(path):
    """Load dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    texts = [format_example(ex) for ex in data]
    return Dataset.from_dict({"text": texts})


def main():
    print("=" * 60)
    print("CricketMind QLoRA Fine-Tuning")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU (RunPod A100).")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Try primary model, fallback if needed
    model_name = PRIMARY_MODEL
    try:
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        model_name = FALLBACK_MODEL
        print(f"Falling back to: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset_from_json(TRAIN_PATH)
    val_dataset = load_dataset_from_json(VAL_PATH)
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete! Next: python scripts/05_merge_and_export.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

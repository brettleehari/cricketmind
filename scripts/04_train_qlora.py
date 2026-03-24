#!/usr/bin/env python3
"""QLoRA fine-tuning script for RunPod A100. DO NOT RUN LOCALLY — requires GPU."""

import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

PRIMARY_MODEL = "nvidia/Nemotron-Mini-4B-Instruct"
TRAIN_PATH = "data/train.json"
VAL_PATH = "data/val.json"
OUTPUT_DIR = "./cricketmind-lora"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]


def format_example(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def load_dataset_from_json(path):
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset_from_json(TRAIN_PATH)
    val_ds = load_dataset_from_json(VAL_PATH)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Load tokenizer
    print(f"\nLoading model: {PRIMARY_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in bfloat16 — skip 4-bit quantization to avoid
    # set_submodule incompatibility with NemotronForCausalLM.
    # 80GB A100 has plenty of VRAM for full bf16.
    # Use dtype for transformers>=5.x, torch_dtype for older versions.
    import transformers
    load_kwargs = dict(
        pretrained_model_name_or_path=PRIMARY_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    if int(transformers.__version__.split(".")[0]) >= 5:
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.config.use_cache = False
    print("Model loaded successfully")

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments — bf16 to match model dtype
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
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
        max_grad_norm=0.3,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"DONE — LoRA adapters saved to {OUTPUT_DIR}")
    print("Next: python scripts/05_merge_and_export.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

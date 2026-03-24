#!/usr/bin/env python3
"""Merge LoRA adapters into base model and export to GGUF. Run on RunPod after training."""

import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PRIMARY_MODEL = "nvidia/Nemotron-Mini-4B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR = "./cricketmind-lora"
MERGED_DIR = "./cricketmind-merged"
GGUF_OUTPUT = "./cricketmind-q4.gguf"


def detect_base_model():
    """Detect which base model was used from adapter config."""
    import json

    config_path = os.path.join(LORA_DIR, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path", PRIMARY_MODEL)
        print(f"Detected base model: {base}")
        return base
    print(f"No adapter config found, defaulting to {PRIMARY_MODEL}")
    return PRIMARY_MODEL


def merge_model():
    """Merge LoRA adapters with base model on CPU."""
    base_model_name = detect_base_model()

    print(f"Loading base model: {base_model_name} (CPU, bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)

    print("Merging adapters...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print("Merge complete!")


def convert_to_gguf():
    """Convert merged model to GGUF q4_k_m format using llama.cpp."""
    llama_cpp_dir = "./llama.cpp"

    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
            check=True,
        )

    # Install conversion dependencies
    print("Installing llama.cpp Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", f"{llama_cpp_dir}/requirements.txt"],
        check=True,
        capture_output=True,
    )

    # Convert to GGUF
    print("Converting to GGUF format...")
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run(
        [
            sys.executable,
            convert_script,
            MERGED_DIR,
            "--outfile",
            GGUF_OUTPUT.replace("-q4.gguf", "-f16.gguf"),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    # Quantize to q4_k_m
    print("Building llama.cpp quantize tool...")
    subprocess.run(["make", "-C", llama_cpp_dir, "llama-quantize"], check=True)

    print("Quantizing to q4_k_m...")
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    subprocess.run(
        [
            quantize_bin,
            GGUF_OUTPUT.replace("-q4.gguf", "-f16.gguf"),
            GGUF_OUTPUT,
            "q4_k_m",
        ],
        check=True,
    )

    # Cleanup f16
    f16_path = GGUF_OUTPUT.replace("-q4.gguf", "-f16.gguf")
    if os.path.exists(f16_path):
        os.remove(f16_path)

    print(f"\nGGUF file saved to: {GGUF_OUTPUT}")
    file_size = os.path.getsize(GGUF_OUTPUT) / (1024 ** 3)
    print(f"File size: {file_size:.2f} GB")


def main():
    print("=" * 60)
    print("CricketMind — Merge & Export")
    print("=" * 60)

    merge_model()
    convert_to_gguf()

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Merged model: {MERGED_DIR}/")
    print(f"  GGUF (q4_k_m): {GGUF_OUTPUT}")
    print("\nNext steps:")
    print("  1. Download cricketmind-q4.gguf to local machine")
    print("  2. STOP your RunPod instance!")
    print("  3. Run: python scripts/07_upload_to_hf.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

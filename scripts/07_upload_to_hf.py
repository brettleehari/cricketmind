#!/usr/bin/env python3
"""Upload model, GGUF, and dataset to Hugging Face. Human runs after training."""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "brettleehari")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add it to .env")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

MODEL_REPO = f"{HF_USERNAME}/cricketmind-nemotron-mini"
DATASET_REPO = f"{HF_USERNAME}/cricketbench-v1"


def upload_model():
    """Upload merged model and GGUF."""
    print(f"Creating model repo: {MODEL_REPO}")
    create_repo(MODEL_REPO, token=HF_TOKEN, exist_ok=True)

    # Upload merged model
    merged_dir = "./cricketmind-merged"
    if os.path.exists(merged_dir):
        print(f"Uploading merged model from {merged_dir}...")
        api.upload_folder(
            folder_path=merged_dir,
            repo_id=MODEL_REPO,
            commit_message="Upload CricketMind merged model",
        )
    else:
        print(f"WARNING: {merged_dir} not found. Run training first.")

    # Upload GGUF
    gguf_path = "./cricketmind-q4.gguf"
    if os.path.exists(gguf_path):
        print(f"Uploading GGUF: {gguf_path}")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="cricketmind-q4.gguf",
            repo_id=MODEL_REPO,
            commit_message="Upload GGUF q4_k_m quantized model",
        )
    else:
        print(f"WARNING: {gguf_path} not found. Run merge & export first.")

    print(f"Model repo: https://huggingface.co/{MODEL_REPO}")


def upload_dataset():
    """Upload training data and evaluation suite."""
    print(f"\nCreating dataset repo: {DATASET_REPO}")
    create_repo(DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

    files_to_upload = [
        ("data/train.json", "train.json"),
        ("data/val.json", "val.json"),
        ("evaluation/cricketbench_v01.json", "cricketbench_v01.json"),
        ("evaluation/judge_results.json", "judge_results.json"),
        ("evaluation/scores_summary.json", "scores_summary.json"),
    ]

    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"  Uploading {local_path} → {repo_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Upload {repo_path}",
            )
        else:
            print(f"  WARNING: {local_path} not found, skipping")

    print(f"Dataset repo: https://huggingface.co/datasets/{DATASET_REPO}")


def main():
    print("=" * 60)
    print("CricketMind — Hugging Face Upload")
    print("=" * 60)

    upload_model()
    upload_dataset()

    print("\n" + "=" * 60)
    print("Upload complete!")
    print(f"  Model: https://huggingface.co/{MODEL_REPO}")
    print(f"  Dataset: https://huggingface.co/datasets/{DATASET_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()

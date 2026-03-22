#!/usr/bin/env python3
"""Combine raw_laws_qa.json + distilled_match_situations.json into train/val splits."""

import json
import random

LAWS_QA_PATH = "data/raw_laws_qa.json"
DISTILLED_PATH = "data/distilled_match_situations.json"
TRAIN_PATH = "data/train.json"
VAL_PATH = "data/val.json"


def main():
    with open(LAWS_QA_PATH) as f:
        laws_qa = json.load(f)
    with open(DISTILLED_PATH) as f:
        distilled = json.load(f)

    # Strip metadata, keep only instruction/input/output
    combined = []
    for ex in laws_qa + distilled:
        combined.append(
            {
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            }
        )

    # Shuffle with seed
    random.seed(42)
    random.shuffle(combined)

    # 90/10 split
    split_idx = int(len(combined) * 0.9)
    train = combined[:split_idx]
    val = combined[split_idx:]

    with open(TRAIN_PATH, "w") as f:
        json.dump(train, f, indent=2)
    with open(VAL_PATH, "w") as f:
        json.dump(val, f, indent=2)

    print(f"Combined: {len(combined)} examples")
    print(f"  Laws QA: {len(laws_qa)}")
    print(f"  Distilled: {len(distilled)}")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"Saved to {TRAIN_PATH} and {VAL_PATH}")


if __name__ == "__main__":
    main()

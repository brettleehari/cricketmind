#!/usr/bin/env python3
"""Generate ~160 QA pairs from MCC Laws using Claude."""

import json
import os
import time
import sys

from dotenv import load_dotenv
import anthropic

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

client = anthropic.Anthropic(api_key=API_KEY)
MODEL = "claude-sonnet-4-20250514"

LAWS_PATH = "data/laws/mcc_laws_text.json"
OUTPUT_PATH = "data/raw_laws_qa.json"


def generate_qa_for_law(law, max_attempts=3):
    """Generate QA pairs for a single law with retries."""
    law_text = json.dumps(law, indent=2)
    prompt = f"""Generate 20 training examples from this cricket law as a JSON array:

{law_text}

Format each example as:
[{{"instruction": "You are CricketMind, an expert cricket analyst and Laws of Cricket specialist. Answer the following question accurately, citing relevant Law numbers and reasoning step by step.", "input": "<question>", "output": "<detailed answer with numbered steps>"}}]

Generate 5 each of:
- Direct recall: straightforward factual questions about the law
- Scenario reasoning: apply the law to a specific game situation
- Misconception: address common misunderstandings about the law
- Edge case: tricky or unusual applications of the law

Rules:
- Numbered steps required in output (use 1. 2. 3. etc.)
- Minimum 150 characters in each output
- Do not hallucinate law numbers — only cite Law {law['number']}
- Return ONLY valid JSON array, no other text"""

    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            examples = json.loads(text)
            return examples
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Attempt {attempt + 1} failed for Law {law['number']}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
    return []


def quality_filter(examples):
    """Keep examples where output >= 150 chars AND contains numbered steps."""
    filtered = []
    for ex in examples:
        output = ex.get("output", "")
        if len(output) >= 150 and any(f"{i}." in output for i in range(1, 10)):
            filtered.append(ex)
    return filtered


def main():
    with open(LAWS_PATH) as f:
        laws_data = json.load(f)

    all_examples = []

    for law in laws_data["laws"]:
        print(f"Generating QA for Law {law['number']}: {law['title']}...")
        examples = generate_qa_for_law(law)
        filtered = quality_filter(examples)
        print(f"  Generated {len(examples)}, kept {len(filtered)} after quality filter")
        all_examples.extend(filtered)
        time.sleep(0.8)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_examples, f, indent=2)

    print(f"\nTotal QA pairs generated: {len(all_examples)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

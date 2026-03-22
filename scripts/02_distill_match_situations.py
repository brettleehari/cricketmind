#!/usr/bin/env python3
"""Response distillation: Claude as teacher generates reasoning traces for match situations."""

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

INPUT_PATH = "data/match_situation_inputs.json"
OUTPUT_PATH = "data/distilled_match_situations.json"

SYSTEM_PROMPT = """You are a world-class cricket analyst and MCC rules official.
Respond in EXACTLY this structure:
1. SITUATION ASSESSMENT: Key variables
2. APPLICABLE LAWS: Cite Law numbers (N/A if strategic)
3. REASONING CHAIN: Step-by-step logic
4. DECISION / RECOMMENDATION: Clear answer
5. CONFIDENCE LEVEL: High/Medium/Low — one sentence why
Under 500 words. No hallucinated law numbers."""

SECTION_HEADERS = [
    "SITUATION ASSESSMENT",
    "APPLICABLE LAWS",
    "REASONING CHAIN",
    "DECISION",
    "CONFIDENCE LEVEL",
]


def distill_scenario(scenario, max_attempts=3):
    """Get Claude's reasoning trace for a scenario."""
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": scenario}],
            )
            output = response.content[0].text.strip()
            return output
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
    return None


def quality_check(output):
    """Reject if too short or missing most section headers."""
    if not output or len(output) < 200:
        return False
    headers_found = sum(1 for h in SECTION_HEADERS if h in output)
    return headers_found >= 3


def main():
    with open(INPUT_PATH) as f:
        scenarios = json.load(f)

    results = []
    rejected = 0

    for i, scenario in enumerate(scenarios):
        print(f"Distilling scenario {i + 1}/{len(scenarios)}...")
        output = distill_scenario(scenario)

        if quality_check(output):
            results.append(
                {
                    "instruction": "You are CricketMind, an expert cricket analyst and Laws of Cricket specialist. Analyze the following scenario step by step.",
                    "input": scenario,
                    "output": output,
                    "type": "distilled_match_situation",
                }
            )
        else:
            rejected += 1
            print(f"  REJECTED (quality check failed)")

        time.sleep(0.6)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDistilled {len(results)} scenarios, rejected {rejected}")
    print(f"Rejection rate: {rejected / len(scenarios) * 100:.1f}%")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

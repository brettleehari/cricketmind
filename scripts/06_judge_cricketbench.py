#!/usr/bin/env python3
"""Fully automated LLM-as-judge evaluation using CricketBench."""

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

BENCH_PATH = "evaluation/cricketbench_v01.json"
RESULTS_PATH = "evaluation/judge_results.json"
SCORES_PATH = "evaluation/scores_summary.json"

STUDENT_SYSTEM = """You are CricketMind, an expert in the Laws of Cricket and match situation analysis.
When answering rules questions: cite Law number, reason step by step, state confidence if ambiguous.
When answering match situation questions: identify key variables first, work through strategic options, give clear recommendation.
Never hallucinate Law numbers. If uncertain, say so explicitly."""

JUDGE_PROMPT_TEMPLATE = """You are an expert cricket judge evaluating a model's response.

QUESTION: {question}

REFERENCE ANSWER: {answer_key}

KEY CONCEPTS THAT SHOULD BE COVERED: {key_concepts}

MODEL RESPONSE: {response}

Score the model response 0-2:
2=fully correct (right answer + reasoning + key concepts)
1=partially correct (right conclusion but weak reasoning OR minor error)
0=incorrect

Respond ONLY with JSON: {{"score": <0|1|2>, "reason": "<one sentence>", "missing": "<what was wrong or 'nothing'>"}}"""

CATEGORY_WEIGHTS = {
    "laws_recall": 0.30,
    "conditional_reasoning": 0.35,
    "match_situation": 0.25,
    "edge_case": 0.10,
}

BASELINE = {
    "laws_recall": 40,
    "conditional_reasoning": 25,
    "match_situation": 30,
    "edge_case": 20,
}


def get_student_response(question):
    """Get CricketMind's response to a question."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=STUDENT_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Student error: {e}")
        return f"Error generating response: {e}"


def judge_response(question, answer_key, key_concepts, response):
    """Judge the student response."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer_key=answer_key,
        key_concepts=", ".join(key_concepts),
        response=response,
    )
    try:
        result = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"score": 0, "reason": f"Judge error: {e}", "missing": "evaluation failed"}


def main():
    with open(BENCH_PATH) as f:
        questions = json.load(f)

    results = []
    category_scores = {}

    print("=" * 60)
    print("CricketBench v0.1 — LLM-as-Judge Evaluation")
    print("=" * 60)

    for i, q in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {q['id']} ({q['category']}) — {q['difficulty']}")
        print(f"  Q: {q['question'][:80]}...")

        # Step 1: Student response
        student_resp = get_student_response(q["question"])
        time.sleep(0.8)

        # Step 2: Judge
        judgment = judge_response(
            q["question"], q["answer_key"], q["key_concepts"], student_resp
        )
        time.sleep(0.8)

        score = judgment.get("score", 0)
        print(f"  Score: {score}/2 — {judgment.get('reason', 'N/A')}")

        results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "student_response": student_resp,
                "judgment": judgment,
                "score": score,
            }
        )

        # Track by category
        if q["category"] not in category_scores:
            category_scores[q["category"]] = []
        category_scores[q["category"]].append(score)

    # Calculate percentages
    cricketmind_scores = {}
    for cat, scores in category_scores.items():
        max_possible = len(scores) * 2
        pct = (sum(scores) / max_possible) * 100 if max_possible > 0 else 0
        cricketmind_scores[cat] = round(pct, 1)

    # Overall weighted score
    overall = sum(
        cricketmind_scores.get(cat, 0) * weight
        for cat, weight in CATEGORY_WEIGHTS.items()
    )
    cricketmind_scores["overall"] = round(overall, 1)

    baseline_overall = sum(
        BASELINE.get(cat, 0) * weight for cat, weight in CATEGORY_WEIGHTS.items()
    )

    summary = {
        "cricketmind": cricketmind_scores,
        "baseline_nemotron": {**BASELINE, "overall": round(baseline_overall, 1)},
    }

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(SCORES_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 60)
    print("CricketBench v0.1 Results")
    print("=" * 60)
    print(f"{'Category':<25} {'CricketMind':>12} {'Baseline':>12} {'Δ':>8}")
    print("-" * 60)
    for cat in CATEGORY_WEIGHTS:
        cm = cricketmind_scores.get(cat, 0)
        bl = BASELINE.get(cat, 0)
        delta = cm - bl
        print(f"{cat:<25} {cm:>11.1f}% {bl:>11.1f}% {delta:>+7.1f}%")
    print("-" * 60)
    cm_o = cricketmind_scores["overall"]
    bl_o = summary["baseline_nemotron"]["overall"]
    print(f"{'OVERALL (weighted)':<25} {cm_o:>11.1f}% {bl_o:>11.1f}% {cm_o - bl_o:>+7.1f}%")
    print("=" * 60)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Scores saved to {SCORES_PATH}")


if __name__ == "__main__":
    main()

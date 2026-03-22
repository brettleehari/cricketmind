#!/usr/bin/env python3
"""Manual expert review of CricketBench LLM-as-judge results.

Walk through each question, see the student response and judge score,
then accept or override with your own score and reasoning.
"""

import json
import os
import sys

RESULTS_PATH = "evaluation/judge_results.json"
REVIEW_PATH = "evaluation/expert_review.json"
SCORES_PATH = "evaluation/scores_summary.json"

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


def print_separator():
    print("\n" + "=" * 70)


def print_wrapped(text, indent=0):
    """Print text with basic wrapping."""
    prefix = " " * indent
    for line in text.split("\n"):
        print(f"{prefix}{line}")


def load_answer_keys():
    """Load reference answers from cricketbench."""
    bench_path = "evaluation/cricketbench_v01.json"
    with open(bench_path) as f:
        questions = json.load(f)
    return {q["id"]: q for q in questions}


def review():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    answer_keys = load_answer_keys()

    # Load existing review if resuming
    reviewed = []
    start_idx = 0
    if os.path.exists(REVIEW_PATH):
        with open(REVIEW_PATH) as f:
            reviewed = json.load(f)
        if reviewed:
            print(f"\nFound {len(reviewed)} previously reviewed questions.")
            resp = input("Resume from where you left off? (y/n): ").strip().lower()
            if resp == "y":
                start_idx = len(reviewed)
            else:
                reviewed = []
                start_idx = 0

    print_separator()
    print("  CricketBench v0.1 — Expert Manual Review")
    print("  Score each response 0-2:")
    print("    2 = Fully correct (right answer + reasoning + key concepts)")
    print("    1 = Partially correct (right conclusion but weak reasoning or minor error)")
    print("    0 = Incorrect (wrong answer or fundamentally flawed)")
    print("  Press Enter to accept the LLM judge's score")
    print("  Type 'q' to save progress and quit")
    print_separator()

    for i in range(start_idx, len(results)):
        r = results[i]
        ref = answer_keys.get(r["id"], {})

        print_separator()
        print(f"  [{i + 1}/{len(results)}]  {r['id']}  |  {r['category']}  |  {r['difficulty']}")
        print_separator()

        print(f"\nQUESTION:")
        print_wrapped(r["question"], indent=2)

        print(f"\nREFERENCE ANSWER:")
        print_wrapped(ref.get("answer_key", "N/A"), indent=2)

        print(f"\nKEY CONCEPTS: {', '.join(ref.get('key_concepts', []))}")

        print(f"\nSTUDENT RESPONSE:")
        print_wrapped(r["student_response"], indent=2)

        print(f"\nLLM JUDGE SCORE: {r['score']}/2")
        print(f"  Reason: {r['judgment'].get('reason', 'N/A')}")
        print(f"  Missing: {r['judgment'].get('missing', 'N/A')}")

        print()
        while True:
            user_input = input(f"Your score (0/1/2) [Enter={r['score']}] or 'q' to quit: ").strip()

            if user_input.lower() == "q":
                # Save progress
                with open(REVIEW_PATH, "w") as f:
                    json.dump(reviewed, f, indent=2)
                print(f"\nProgress saved ({len(reviewed)}/{len(results)} reviewed).")
                print(f"Run this script again to resume.")
                return

            if user_input == "":
                expert_score = r["score"]
                expert_reason = ""
                break
            elif user_input in ("0", "1", "2"):
                expert_score = int(user_input)
                if expert_score != r["score"]:
                    expert_reason = input("Your reasoning for the override: ").strip()
                else:
                    expert_reason = ""
                break
            else:
                print("  Please enter 0, 1, 2, Enter to accept, or 'q' to quit.")

        reviewed.append({
            "id": r["id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "question": r["question"],
            "student_response": r["student_response"],
            "llm_judge_score": r["score"],
            "llm_judge_reason": r["judgment"].get("reason", ""),
            "expert_score": expert_score,
            "expert_reason": expert_reason,
            "overridden": expert_score != r["score"],
        })

        status = "ACCEPTED" if expert_score == r["score"] else f"OVERRIDDEN: {r['score']} → {expert_score}"
        print(f"  → {status}")

    # All reviewed — save and recalculate
    with open(REVIEW_PATH, "w") as f:
        json.dump(reviewed, f, indent=2)

    # Recalculate scores
    category_scores = {}
    overrides = 0
    for r in reviewed:
        cat = r["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(r["expert_score"])
        if r["overridden"]:
            overrides += 1

    cricketmind_scores = {}
    for cat, scores in category_scores.items():
        max_possible = len(scores) * 2
        pct = (sum(scores) / max_possible) * 100 if max_possible > 0 else 0
        cricketmind_scores[cat] = round(pct, 1)

    overall = sum(
        cricketmind_scores.get(cat, 0) * weight
        for cat, weight in CATEGORY_WEIGHTS.items()
    )
    cricketmind_scores["overall"] = round(overall, 1)

    baseline_overall = sum(
        BASELINE.get(cat, 0) * weight for cat, weight in CATEGORY_WEIGHTS.items()
    )

    summary = {
        "cricketmind_expert_reviewed": cricketmind_scores,
        "cricketmind_llm_judge": {},  # preserve original
        "baseline_nemotron": {**BASELINE, "overall": round(baseline_overall, 1)},
        "review_stats": {
            "total_questions": len(reviewed),
            "overrides": overrides,
            "override_rate": f"{overrides / len(reviewed) * 100:.1f}%",
        },
    }

    # Load original LLM judge scores
    with open(SCORES_PATH) as f:
        original = json.load(f)
    summary["cricketmind_llm_judge"] = original.get("cricketmind", {})

    with open(SCORES_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final comparison
    print_separator()
    print("  EXPERT REVIEW COMPLETE")
    print_separator()
    print(f"\n  Overrides: {overrides}/{len(reviewed)} ({overrides / len(reviewed) * 100:.1f}%)")

    print(f"\n{'Category':<25} {'Expert':>10} {'LLM Judge':>10} {'Baseline':>10}")
    print("-" * 60)
    for cat in CATEGORY_WEIGHTS:
        ex = cricketmind_scores.get(cat, 0)
        lj = summary["cricketmind_llm_judge"].get(cat, 0)
        bl = BASELINE.get(cat, 0)
        print(f"{cat:<25} {ex:>9.1f}% {lj:>9.1f}% {bl:>9.1f}%")
    print("-" * 60)
    ex_o = cricketmind_scores["overall"]
    lj_o = summary["cricketmind_llm_judge"].get("overall", 0)
    bl_o = summary["baseline_nemotron"]["overall"]
    print(f"{'OVERALL (weighted)':<25} {ex_o:>9.1f}% {lj_o:>9.1f}% {bl_o:>9.1f}%")
    print_separator()

    print(f"\nExpert review saved to: {REVIEW_PATH}")
    print(f"Updated scores saved to: {SCORES_PATH}")


if __name__ == "__main__":
    review()

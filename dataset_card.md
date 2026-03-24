---
language:
- en
license: apache-2.0
tags:
- cricket
- laws-of-cricket
- domain-adaptation
- evaluation
size_categories:
- n<1K
---

# CricketBench v0.1

Evaluation suite and training data for cricket domain adaptation.

## Contents

- **train.json** — 153 training examples (Laws QA + distilled match situations)
- **val.json** — 17 validation examples
- **cricketbench_v01.json** — 20 evaluation questions across 4 categories
- **judge_results.json** — Full LLM-as-judge evaluation results
- **scores_summary.json** — Aggregated scores

## Categories

| Category | Weight | Questions |
|---|---|---|
| Laws Recall | 30% | 5 |
| Conditional Reasoning | 35% | 5 |
| Match Situation | 25% | 5 |
| Edge Case | 10% | 5 |

## Associated Model

[brettleehari/cricketmind-nemotron-mini](https://huggingface.co/brettleehari/cricketmind-nemotron-mini)

## Author

**Hariprasad Sudharshan** — [LinkedIn](https://linkedin.com/in/haripm4ai) | [GitHub](https://github.com/brettleehari)

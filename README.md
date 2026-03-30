# CricketMind

**Fine-tuning NVIDIA Nemotron-Mini-4B for cricket domain expertise**

CricketMind is an end-to-end fine-tuning project that transforms a general-purpose LLM into a cricket laws and match-situation reasoning specialist. It shows that meaningful domain adaptation is possible with ~170 training examples, 35 minutes of GPU time, and under $5.

## Results

| Metric | Baseline (Nemotron-Mini-4B) | Fine-Tuned | Improvement |
|--------|----------------------------|------------|-------------|
| Overall Score | 29.8% | 67.5% | **+37.7 pp** |

Evaluated on **CricketBench v0.1** — a custom benchmark with 20 questions across 4 difficulty tiers, scored using LLM-as-judge methodology.

## What's Inside

### Data Pipeline
- **Laws Corpus:** 8 major MCC cricket laws -> ~160 generated QA pairs
- **Match Situations:** 50 real-world scenarios processed through response distillation (Claude as teacher model)
- **Evaluation Set:** CricketBench — 20 questions, 4 difficulty categories, weighted scoring

### Training
- **Method:** QLoRA fine-tuning (rank 16, 4-bit quantization)
- **Precision:** bfloat16
- **Hardware:** RunPod A100 (~$4 total cost)
- **Time:** ~35 minutes

### Project Structure

```
cricketmind/
 data/             # Training data and laws corpus
 evaluation/       # CricketBench benchmark and scoring
 model/            # Model configs and training scripts
 scripts/          # Data generation and pipeline utilities
 model_card.md     # Full model documentation
 dataset_card.md   # Dataset documentation
```



## Key Techniques

- **Response Distillation** — Using Claude as teacher to generate high-quality training data
- **QLoRA** — Parameter-efficient fine-tuning with 4-bit quantisation for affordable GPU training
- **LLM-as-Judge Evaluation** — Automated scoring with confidence thresholds (0.8) and structured rubrics
- **Custom Benchmarking** — Domain-specific eval with weighted difficulty tiers

## Quickstart

```bash
git clone https://github.com/brettleehari/cricketmind.git
cd cricketmind
pip install -r requirements.txt
python scripts/generate_qa_pairs.py
python evaluation/run_benchmark.py
```

Fine-tuning requires GPU access (A100 recommended). S

## Author

**Hariprasad Sudharshan** - [GitHub](https://github.com/brettleehari) - OSS contributor to [LlamaIndex](https://github.com/run-llama/llama_index/pull/15311)

## License

MIT

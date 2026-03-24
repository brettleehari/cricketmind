---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- cricket
- domain-adaptation
- qlora
- lora
- nemotron
- sports
- rules-engine
datasets:
- brettleehari/cricketbench-v1
base_model: nvidia/Nemotron-Mini-4B-Instruct
pipeline_tag: text-generation
model-index:
- name: CricketMind Nemotron Mini
  results:
  - task:
      type: text-generation
      name: Cricket Domain QA
    dataset:
      name: CricketBench v0.1
      type: brettleehari/cricketbench-v1
    metrics:
    - type: accuracy
      value: 67.5
      name: Overall CricketBench Score
---

# CricketMind — Cricket Domain Expert (Nemotron Mini 4B)

A fine-tuned version of [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) specialized in **MCC Laws of Cricket** and **match situation analysis**.

## Training

- **Method**: LoRA (r=16, alpha=32) on bfloat16
- **Target modules**: q_proj, v_proj
- **Data**: ~170 examples — Laws QA + response distillation from Claude
- **Hardware**: NVIDIA A100 80GB SXM
- **Epochs**: 3
- **Final training loss**: 1.65

## Evaluation — CricketBench v0.1

LLM-as-judge evaluation (Claude) across 20 questions in 4 categories:

| Category | CricketMind | Baseline Nemotron | Improvement |
|---|---|---|---|
| Laws Recall (30%) | 60.0% | 40% | +20pp |
| Conditional Reasoning (35%) | 70.0% | 25% | +45pp |
| Match Situation (25%) | 80.0% | 30% | +50pp |
| Edge Case (10%) | 50.0% | 20% | +30pp |
| **Overall** | **67.5%** | **30.2%** | **+37.3pp** |

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "brettleehari/cricketmind-nemotron-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = """### Instruction:
You are CricketMind, an expert in the Laws of Cricket. Cite Law numbers and reason step by step.

### Input:
A batter is struck on the pad outside the line of off stump. They played a shot. Is it out LBW?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Google Colab (free GPU)

1. Go to [colab.google.com](https://colab.google.com) → New notebook
2. Runtime → Change runtime type → T4 GPU
3. Paste the code above and run

## Dataset

Training data and evaluation suite: [brettleehari/cricketbench-v1](https://huggingface.co/datasets/brettleehari/cricketbench-v1)

## Author

**Hariprasad Sudharshan** — AI Product Manager
- [LinkedIn](https://linkedin.com/in/haripm4ai)
- [GitHub](https://github.com/brettleehari)

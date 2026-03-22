# Fine-Tuning Nemotron for Cricket Domain Adaptation: A PM's Technical Deep Dive

**How I built CricketMind — a domain-expert model for cricket rules and match reasoning — using QLoRA, response distillation, and NVIDIA's Nemotron**

*By Hariprasad Sudharshan*

---

## 1. Why Cricket. Why Now.

Cricket isn't just a sport — it's a rule system of extraordinary complexity. The MCC Laws of Cricket span 42 laws with hundreds of edge cases, conditional logic branches, and format-specific interpretations that differ between Test, ODI, and T20 cricket. Consider LBW alone: three spatial conditions must hold simultaneously, with a no-shot exception that overrides one condition, an absolute leg-side exception that overrides everything, and a DRS layer that introduces probabilistic "Umpire's Call" zones.

This makes cricket an ideal domain adaptation benchmark. If a small language model can learn to reason about LBW conditions, DLS recalculations, and ball tampering penalties with step-by-step logic, then domain adaptation via fine-tuning is working — not just memorization, but genuine conditional reasoning over structured rules.

The question I wanted to answer: **Can a 4B parameter model, fine-tuned with fewer than 200 examples, outperform its base version on domain-specific conditional reasoning?**

---

## 2. Choosing the Right Base: Why Nemotron

I chose NVIDIA's Nemotron-Mini-4B-Instruct as the base model for three reasons:

1. **Reasoning depth at small scale.** Nemotron-Mini-4B punches above its weight on structured reasoning tasks compared to similarly-sized models. For a domain that requires multi-step conditional logic (pitch → impact → hitting stumps, each with exceptions), reasoning capability matters more than raw knowledge.

2. **Context window.** The 4096-token context window is sufficient for the structured reasoning traces I'm distilling (under 500 words each), while keeping inference fast enough for interactive use.

3. **NVIDIA ecosystem alignment.** Nemotron sits in the NVIDIA AI stack — TensorRT-LLM, NIM microservices, Triton Inference Server. A fine-tuned Nemotron can be deployed through the same enterprise pipeline that organizations already use for NVIDIA models. This isn't an academic exercise; it's a deployable architecture.

---

## 3. The Dataset Problem

The hardest part of this project wasn't training — it was data. Three layers of data generation, each with its own quality gates:

### Layer 1: Laws Corpus QA (~120 pairs)
I extracted 8 core MCC Laws and generated 20 QA pairs per law using Claude, covering four question types:
- **Direct recall** — "What are the three LBW conditions?"
- **Scenario reasoning** — "Ball pitches on middle, hits pad outside off, batter played a shot. Out?"
- **Misconception correction** — "Can you be LBW to a ball pitching outside leg?"
- **Edge cases** — "What if the bail doesn't fall when the ball hits the stumps?"

**Quality gate:** Every answer must contain numbered reasoning steps and exceed 150 characters. This filter rejected ~10% of generated examples that were too terse or lacked structured reasoning.

### Layer 2: Response Distillation (50 match situations)
This is where the real value lives. I created 50 complex match scenarios — from "18 off the last over with 3 wickets remaining" to "DLS par score implications after rain interruption" — and used Claude as a teacher model to generate structured reasoning traces.

### Layer 3: Combine and Split
The final dataset: **170 examples** total (120 laws QA + 50 distilled situations), shuffled and split 90/10 into 153 training and 17 validation examples.

---

## 4. Distillation Methodology

Response distillation is the core technique. Rather than having the student model learn from raw question-answer pairs, it learns from a teacher model's structured reasoning process.

I used Claude as the teacher model — both for its superior structured reasoning on rules-based conditional logic, and for practical efficiency given my existing Claude Pro access.

### The 5-Part Response Format

Every teacher response follows this exact structure:

```
1. SITUATION ASSESSMENT: Key variables affecting the decision
2. APPLICABLE LAWS: Specific Law numbers cited (N/A if purely strategic)
3. REASONING CHAIN: Step-by-step logic working through the problem
4. DECISION / RECOMMENDATION: Clear, unambiguous answer
5. CONFIDENCE LEVEL: High/Medium/Low with one-sentence justification
```

This format forces the teacher model to externalize its reasoning chain. When the student model learns to replicate this structure, it's not just learning answers — it's learning *how to reason about cricket problems*.

### Quality Gate

The distillation quality gate rejected any response that:
- Was under 200 characters (too terse to contain meaningful reasoning)
- Missing 3 or more of the 5 section headers (unstructured)

In practice, the rejection rate was **0%** — Claude consistently produced well-structured responses. This speaks to the reliability of using a frontier model as a teacher: the quality variance is low enough that aggressive filtering isn't needed.

### Why Not GPT-4o?

Claude's structured reasoning on conditional logic — particularly the kind of "if X AND Y but NOT Z, then..." reasoning that cricket laws demand — was noticeably stronger in my testing. The 5-part format was consistently followed without prompt engineering gymnastics. For a distillation pipeline where consistency matters as much as correctness, this reliability is critical.

---

## 5. Fine-Tuning Methodology: QLoRA

The training pipeline combines three techniques: 4-bit quantization, Low-Rank Adaptation, and the distilled dataset.

### Quantization: NF4

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # Normal Float 4-bit
    bnb_4bit_compute_dtype=bfloat16,  # Compute in bfloat16
    bnb_4bit_use_double_quant=True,   # Quantize the quantization constants
)
```

NF4 (Normal Float 4-bit) is information-theoretically optimal for normally distributed weights. Double quantization further reduces memory by quantizing the quantization constants themselves. This brings the 4B parameter model's memory footprint from ~8GB (fp16) to ~2.5GB, making A100 40GB massively overprovisioned — but that headroom is useful for larger batch sizes.

### LoRA: Rank 16

```python
LoraConfig(
    r=16,              # Rank — balance between capacity and efficiency
    lora_alpha=32,     # Scaling factor (alpha/r = 2)
    target_modules=["q_proj", "v_proj"],  # Attention projections only
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Why rank 16?** For ~170 training examples, rank 16 provides sufficient capacity to learn domain-specific patterns without overfitting. The rule of thumb: rank should scale with dataset complexity, not dataset size. Cricket's conditional logic is complex enough to warrant r=16 over r=8, but not complex enough (at 170 examples) to justify r=32.

**Why only q_proj and v_proj?** These attention projection matrices control *what the model attends to* (queries) and *what information it extracts* (values). For domain adaptation where the model needs to learn new attention patterns over cricket-specific concepts, these are the highest-leverage targets. Adding k_proj and other modules would increase trainable parameters without proportional benefit at this dataset size.

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch size: 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",
)
```

Three epochs over 153 examples means ~459 training steps with effective batch size 16. The cosine schedule with 3% warmup gives a gentle ramp-up before the main learning phase. Paged AdamW in 8-bit further reduces optimizer state memory.

**Total training time:** ~35 minutes on an A100 40GB. **Total cost:** ~$1.10 at RunPod rates.

---

## 6. CricketBench: Designing the Evaluation Framework

Off-the-shelf benchmarks (MMLU, HellaSwag, ARC) don't test cricket domain knowledge. I built CricketBench v0.1: 20 questions across four dimensions, evaluated by an LLM-as-judge.

### Four Dimensions

| Dimension | Weight | What It Tests |
|---|---|---|
| Laws Recall | 30% | Can the model accurately recall specific law provisions? |
| Conditional Reasoning | 35% | Can it apply conditional logic (if/then/except) to novel scenarios? |
| Match Situation | 25% | Can it synthesize strategy from multiple variables? |
| Edge Cases | 10% | Can it handle unusual rule interactions? |

Conditional reasoning gets the highest weight because it's the hardest capability to acquire and the most valuable in practice. Anyone can memorize "three LBW conditions" — the test is whether the model correctly applies the no-shot exception when asked about a specific scenario.

### LLM-as-Judge Protocol

For each of the 20 questions:
1. **Student phase:** Claude (acting as CricketMind) generates a response
2. **Judge phase:** Claude (acting as judge) scores 0-2 against a reference answer and key concepts

Scoring rubric:
- **2** = Fully correct: right answer + sound reasoning + key concepts covered
- **1** = Partially correct: right conclusion but weak reasoning, or minor factual error
- **0** = Incorrect: wrong answer or fundamentally flawed reasoning

### Results

| Category | CricketMind | Baseline Nemotron | Delta |
|---|---|---|---|
| Laws Recall | **60.0%** | 40.0% | **+20.0%** |
| Conditional Reasoning | **70.0%** | 25.0% | **+45.0%** |
| Match Situation | **80.0%** | 30.0% | **+50.0%** |
| Edge Cases | **50.0%** | 20.0% | **+30.0%** |
| **Overall (weighted)** | **67.5%** | **30.2%** | **+37.3%** |

The largest gains are in conditional reasoning (+45%) and match situation analysis (+50%) — exactly where the distillation methodology should have the most impact. The structured reasoning traces teach the model *how to decompose problems*, not just what the answers are.

Edge cases show the smallest improvement (+30%), which makes sense: edge cases by definition involve unusual rule interactions that may not be well-represented in 170 training examples. This is the clearest signal for where v2 data expansion should focus.

---

## 7. Deploying with Ollama

After training and GGUF export, CricketMind runs locally via Ollama:

```bash
# Import the model
ollama create cricketmind -f model/Modelfile

# Run inference
ollama run cricketmind "A batter is struck on the pad outside off stump.
They played no shot. Ball pitched on middle stump and was hitting leg stump.
Is the batter out LBW?"
```

The Modelfile configures low temperature (0.1) for consistent, deterministic responses on rules questions, with a system prompt that enforces the CricketMind persona:

```
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

The q4_k_m quantization keeps the model under 3GB, runnable on any modern laptop without a GPU.

---

## 8. Publishing to Hugging Face

The complete package is published to Hugging Face:

- **Model:** `brettleehari/cricketmind-nemotron-mini` — merged model + GGUF quantized version
- **Dataset:** `brettleehari/cricketbench-v1` — training data, evaluation suite, and judge results

The model card includes training configuration, intended use cases, limitations, and the full CricketBench results for reproducibility.

---

## 9. What This Project Taught Me

### The data problem dwarfs the training problem

I spent 70% of project time on data generation and quality filtering, 10% on training configuration, and 20% on evaluation design. This ratio is not an accident — it's the reality of applied ML. The model is only as good as its training signal, and for domain-specific tasks, that signal doesn't exist off the shelf.

### Evaluation is a product decision

CricketBench's four dimensions and their weights aren't technical choices — they're product choices. Weighting conditional reasoning at 35% reflects a belief that *reasoning capability* matters more than *recall capability* for a cricket assistant. A different product vision (say, a quiz app) would weight laws recall higher. The evaluation framework encodes your product thesis.

### Confidence gating belongs in the architecture

In my work at Fujitsu, I implemented a 0.8 confidence threshold for automated decisions. The same principle applies here: CricketMind's system prompt instructs it to "state confidence if ambiguous." In production, responses below a confidence threshold should route to human review. This isn't a nice-to-have — it's the difference between a demo and a deployed system.

---

## 10. What's Next

**CricketMind v1 Roadmap:**

- **Dataset expansion:** Target 500+ examples, focusing on edge cases (currently the weakest category at 50%)
- **Format-specific models:** Separate adapters for Test, ODI, and T20 rule interpretations
- **Real-time match integration:** Connect to live scoring APIs for context-aware analysis
- **Multi-turn dialogue:** Support follow-up questions ("What if the batter had played a shot?")
- **Benchmark expansion:** CricketBench v1.0 with 100+ questions and human expert validation

The goal isn't to replace cricket umpires — it's to build a reasoning system that can explain *why* a decision is correct, step by step, citing the relevant laws. That's a product that serves commentators, coaches, fans, and anyone learning the game.

---

*Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications, focused on enterprise AI deployment and domain adaptation. Connect on [LinkedIn](https://linkedin.com/in/haripm4ai) or explore the code on [GitHub](https://github.com/brettleehari).*

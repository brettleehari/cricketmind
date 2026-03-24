# From Weights to Product — The Full AI Value Chain, Walked

**Building CricketMind, Part 6 of 6**

*By Hariprasad Sudharshan*

---

Six articles ago, I set out to answer whether a PM should get their hands dirty at the model layer. The answer is unequivocally yes — but not for the reasons I expected.

I assumed the value would be technical literacy — knowing enough to call bluffs in vendor meetings or push back on engineering timelines. That happened, but the real payoff was something deeper. Walking every step of the AI value chain — from selecting a base model to publishing a fine-tuned one on HuggingFace — rewired how I think about AI product decisions. Not because I became an ML engineer. Because I stopped abstracting away the layers where the hardest tradeoffs live.

This final article recaps the journey, shares the deployment story, breaks down the real costs, and distills the five lessons I will carry into every product decision from here.

---

## The Value Chain, Walked

Over five articles and several weekends, CricketMind moved from idea to published model. Here is what each layer actually taught me.

### 1. Model Selection (Part 2) — Ecosystem Fit Over Benchmarks

Choosing Nemotron-Mini-4B-Instruct over Llama taught me that ecosystem fit matters more than benchmark scores. Nemotron sits inside the NVIDIA deployment stack — TensorRT-LLM, NIM, Triton — which means a fine-tuned version slots into enterprise pipelines without re-architecture. The benchmark delta between Nemotron and Llama-3.2-3B was marginal. The deployment story was not. PMs who evaluate models purely on leaderboard position are optimizing the wrong variable.

### 2. Data (Part 3) — 70% of the Work, 170 Examples

Data generation consumed the majority of the project's engineering time. I used Claude as a teacher model to produce structured reasoning traces across two data layers: 120 laws QA pairs generated from MCC Law summaries, and 50 response-distilled match situation analyses with a mandatory five-part reasoning format. The total dataset was 170 examples. The quality gate — minimum character counts, required section headers, numbered reasoning steps — rejected roughly 10% of generated content. Data quality over quantity is not a platitude. It is the operational reality of fine-tuning with limited examples.

### 3. Training (Part 4) — 37 Seconds of Training, Hours of Debugging

The actual QLoRA training run on an A100 took about 35 minutes for three epochs over 153 examples. The time surrounding it — resolving transformers version incompatibilities with Nemotron's architecture, debugging bitsandbytes CUDA errors, managing disk space on a cloud GPU instance — dwarfed the training itself. Every fine-tuning tutorial makes it look like a single script execution. In practice, the gap between tutorial and reality is filled with version matrices, environment configuration, and library-specific edge cases. This is the knowledge that does not transfer from reading documentation. You have to hit the errors yourself.

### 4. Evaluation (Part 5) — CricketBench Weights Encode Product Thesis

Building CricketBench — 20 questions across four weighted dimensions, scored by an LLM-as-judge — was where product thinking and ML evaluation became the same discipline. The weight I assigned to conditional reasoning (35%) over laws recall (30%) was not a technical decision. It was a product bet: that a cricket assistant's value comes from its ability to reason through novel scenarios, not from memorizing rule text. The final results validated the approach: 67.5% overall accuracy, a +37.3 percentage point improvement over the baseline Nemotron, with the largest gains in conditional reasoning (+45pp) and match situation analysis (+50pp). Those gains map directly to where the distillation methodology applies the most pressure.

### 5. Deployment (This Article) — Published, Accessible, Real

The model is on HuggingFace. The evaluation suite is public. The inference code works. This layer is where the work stops being a learning exercise and becomes a product artifact that others can use, critique, and build on.

---

## Publishing to HuggingFace

The complete CricketMind package is published in two repositories:

- **Model:** [huggingface.co/brettleehari/cricketmind-nemotron-mini](https://huggingface.co/brettleehari/cricketmind-nemotron-mini) — the merged QLoRA fine-tuned model plus a GGUF-quantized version for local inference
- **Dataset:** [huggingface.co/datasets/brettleehari/cricketbench-v1](https://huggingface.co/datasets/brettleehari/cricketbench-v1) — the training data, the CricketBench evaluation suite, and the full judge results

The model card includes the training configuration (QLoRA r=16, NF4 quantization, 3 epochs, 2e-4 learning rate), the CricketBench results table with per-category scores, intended use cases, known limitations, and the base model lineage. Publishing a model without its evaluation results and training provenance is publishing a black box. The model card is not optional — it is the minimum bar for responsible release.

Making the model accessible mattered to me. The repository includes usage examples, the Modelfile for Ollama deployment, and inference code that runs on consumer hardware. A model that only works on an A100 is a research artifact. A model that runs on a laptop is a product.

---

## The Inference Experience

After all the training, merging, and quantization, using CricketMind comes down to a few lines of Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "brettleehari/cricketmind-nemotron-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

prompt = """You are CricketMind, an expert in the Laws of Cricket.
Cite Law numbers, reason step by step.

Question: A batter is struck on the pad outside off stump.
They played no shot. Ball pitched on middle stump and
was hitting leg stump. Is the batter out LBW?"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or, with the GGUF version via Ollama:

```bash
ollama create cricketmind -f model/Modelfile
ollama run cricketmind "Under Law 21, what happens on a No Ball free hit?"
```

The quantized model is under 3GB. No GPU required. This is the deployment story that matters for domain-specific models: small enough to run locally, specialized enough to be useful, accessible enough that anyone with a laptop can try it.

---

## What This Project Cost

Here is the honest accounting:

| Item | Cost |
|---|---|
| RunPod A100 40GB (~2 hours including debugging) | ~$3-4 |
| Claude API for data generation and evaluation | ~$3-4 |
| HuggingFace hosting | Free |
| GitHub repository | Free |
| Ollama | Free |
| **Total** | **Under $10** |

Under ten dollars for a published, fine-tuned, evaluated domain model with a public evaluation benchmark and inference code.

The real cost was engineering time — roughly 40-50 hours across the series, spread over weekends. But that is the point. The knowledge stays with you. The ten dollars buys compute. The hours buy understanding that compounds across every future product decision involving AI.

Three years ago, this project would have required a dedicated ML team, cloud training budgets in the hundreds, and weeks of calendar time. The infrastructure democratization is real, and PMs who do not update their mental models of "what AI costs" are operating on stale assumptions.

---

## Five Things I Now Know That I Did Not Before

**1. Custom model architectures have real compatibility gaps with standard tooling.**

Nemotron's architecture required a specific transformers version that differed from what other libraries expected. This is not a bug report — it is a product insight. When vendors claim "we support all HuggingFace models," I now know to ask which architecture variants, which quantization methods, and which library version matrix they have actually tested. The gap between "technically compatible" and "works without debugging" is where implementation timelines die.

**2. Response distillation from a frontier model is the highest-leverage data strategy for domain adaptation.**

Using Claude as a teacher model to generate structured reasoning traces — not just answers, but externalized chains of thought — produced the largest quality gains per example. The five-part response format (Situation Assessment, Applicable Laws, Reasoning Chain, Decision, Confidence Level) forced the teacher to make its reasoning legible. The student model learned the reasoning pattern, not just the conclusions. I will recommend this approach in product planning for any domain adaptation project. It is faster, cheaper, and more reliable than human annotation at comparable quality.

**3. Evaluation design is product design.**

Building CricketBench taught me more about product thinking than any prioritization framework. Every decision — the four categories, their weights, the scoring rubric, the judge prompt — was a product decision disguised as a technical one. Weighting conditional reasoning at 35% was a statement about what the product values. The evaluation framework does not just measure the model. It encodes what "good" means for your use case. PMs should be designing eval frameworks, not delegating them.

**4. The gap between fine-tuning tutorials and production is wider than most PMs assume.**

Every tutorial shows the clean path. None of them show the transformers version that breaks bitsandbytes, the disk space error at 87% of merge completion, or the tokenizer that silently truncates your prompts. The time estimate for "fine-tune a model" in a product roadmap needs a 3-5x multiplier for environment debugging, version resolution, and edge cases that only surface at execution time. PMs who have never hit these errors will consistently underestimate timelines.

**5. 170 examples can create genuine new capability in a 4B parameter model.**

This fundamentally changes the economics of domain-specific AI. The +37.3 percentage point improvement over baseline came from fewer than 200 carefully constructed examples. Not thousands. Not millions. The implication: for any domain with structured rules and conditional logic — regulatory compliance, medical protocols, legal standards, financial regulations — a small, high-quality dataset with response distillation can produce a meaningfully capable domain model. The barrier is not data volume. It is data design.

---

## What Is Next for CricketMind

CricketMind v0.1 is a proof of concept. The roadmap to v1 is clear, driven directly by what the evaluation revealed:

- **Dataset expansion to 500+ examples**, with focused investment in edge cases — currently the weakest category at 50%. The distillation pipeline is built; scaling it is an API call and a budget decision.
- **Format-specific adapters** for Test, ODI, and T20 rule interpretations. Cricket's laws have format-dependent applications (Wide Ball interpretation is stricter in T20s, for instance) that a single adapter conflates.
- **Multi-turn dialogue** to support follow-up reasoning. "What if the batter had played a shot?" is the natural next question after any LBW analysis, and the current model handles each query independently.
- **CricketBench v1.0** with 100+ questions and human expert validation. Twenty questions was sufficient to surface category-level signal. A hundred will surface question-level reliability.
- **Real-time match integration** via live scoring APIs, enabling context-aware analysis during actual matches. This is the deployment scenario where a domain model becomes a product.

---

## What This Means for PMs

The full value chain knowledge compounds. Understanding how data quality affects training loss affects evaluation design affects deployment decisions — each layer informs every other layer. A PM who has only seen the deployment layer makes different (and worse) decisions than one who has walked the full chain.

You do not need to become an ML engineer. You need to have built one thing end-to-end so you know where the bodies are buried. You need to know that data generation is 70% of the work, that evaluation encodes product values, that version matrices are the real timeline risk, and that 170 examples can move a 4B model by 37 percentage points. You cannot learn these things from reading about them. You learn them by building.

The cost barrier is gone. Under ten dollars and a series of weekends. The compute is cheap. The tooling is mature enough. The frontier models that serve as teachers are accessible via API. The only remaining barrier is the decision to start.

---

## Build Your Own

The code is open source: [github.com/brettleehari/cricketmind](https://github.com/brettleehari/cricketmind). The model is on HuggingFace: [brettleehari/cricketmind-nemotron-mini](https://huggingface.co/brettleehari/cricketmind-nemotron-mini). The evaluation benchmark is public: [brettleehari/cricketbench-v1](https://huggingface.co/datasets/brettleehari/cricketbench-v1).

Pick your domain. Pick your base model. Build.

The methodology transfers. Response distillation works for any domain with structured reasoning. QLoRA runs on rented hardware for single-digit dollars. CricketBench's evaluation framework — weighted categories, LLM-as-judge, baseline comparison — is a template you can adapt to regulatory compliance, medical triage protocols, or whatever domain your product serves.

The question is not whether PMs should understand the model layer. The question is whether you can afford not to.

---

*Thank you for reading this series.*

---

**About the author**

Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications, where he works on enterprise AI deployment and domain adaptation. He built CricketMind to demonstrate that the full AI value chain — from model selection through fine-tuning, evaluation, and deployment — is accessible to product managers willing to get their hands dirty.

GitHub: [github.com/brettleehari](https://github.com/brettleehari)
LinkedIn: [linkedin.com/in/haripm4ai](https://linkedin.com/in/haripm4ai)

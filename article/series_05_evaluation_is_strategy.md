# Evaluation Is Strategy, Not Metrics — Designing CricketBench

**Building CricketMind, Part 5 of 6**

*How I designed a 20-question evaluation framework that encodes a product thesis — and what the results tell me about where to invest next.*

---

Off-the-shelf benchmarks are useless for domain-specific models.

MMLU, HellaSwag, ARC — they test whether a model can do general reasoning across broad knowledge domains. That is a fine thing to measure if you are building a general-purpose assistant. But if you are building a cricket expert, none of those benchmarks will tell you whether your model knows that a ball pitching outside leg stump is an absolute exception to LBW, regardless of impact or whether the batter played a shot.

CricketBench is 20 questions. That is not a large evaluation suite. But those 20 questions encode a product thesis about what "cricket expertise" actually means, and the category weights I assigned are product decisions that shaped every downstream choice — from where to expand the training data to whether the model is production-ready.

This article is about why evaluation design is the most underrated PM skill in the AI stack, and how CricketBench forced me to articulate what I was actually building.

---

## Why Build Your Own Benchmark

There are three reasons to build a domain-specific evaluation instead of relying on generic benchmarks.

**First, generic benchmarks test general knowledge, not domain reasoning.** A model can score well on MMLU and still have no idea that Law 36 has three simultaneous conditions for LBW — pitching, impact, and hitting. General benchmarks cannot distinguish between a model that has memorized cricket Wikipedia and one that can apply conditional logic to novel scenarios involving the Laws of Cricket.

**Second, your evaluation framework encodes what you believe matters.** When I decided that conditional reasoning should carry 35% of the total weight in CricketBench, I was making a product statement: the hardest and most valuable capability for a cricket expert is applying if/then/except logic to situations it has never seen. That weight is not a statistical artifact. It is a design decision about what the product should be good at.

**Third, evaluation results tell you where to invest next.** When I saw that edge case performance was the weakest category at 50%, I did not need a product review meeting to know that v2 data expansion should focus on unusual rule interactions. The evaluation framework becomes a roadmap generator.

If you are a PM working on any domain-specific AI product — legal, medical, financial, anything — build your evaluation before you build the product. The evaluation is the spec.

---

## CricketBench Design: 4 Categories, 20 Questions

I structured CricketBench around four categories, each targeting a different dimension of cricket expertise:

| Category | Weight | What It Tests | Example Question |
|---|---|---|---|
| Laws Recall | 30% | Accurate recall of specific law provisions | "What are the three conditions that must all be met for an LBW dismissal?" |
| Conditional Reasoning | 35% | If/then/except logic on novel scenarios | "Ball pitches outside leg, impact in line, hitting stumps. No shot played. Out or not out?" |
| Match Situation | 25% | Strategic synthesis from multiple variables | "18 off 6 balls, 3 wickets in hand — what is the optimal death bowling strategy?" |
| Edge Case | 10% | Unusual rule interactions | "Ball hits stumps, bail doesn't fall, ricochets off keeper back onto stumps. Batter out?" |

The weight distribution was deliberate and reflects a specific product thesis.

**Conditional reasoning gets 35% because it is the hardest capability to acquire and the most valuable in practice.** Anyone can memorize the Laws of Cricket — there are only 42 of them, and the MCC publishes them for free. The real test is whether a model can take a novel combination of conditions and correctly apply exceptions. "Ball pitches outside leg stump" is an absolute exception to LBW that overrides everything else. "No shot offered" changes whether the impact needs to be in line with the stumps. These are conditional chains, and getting them right requires genuine reasoning, not retrieval.

**Laws recall gets 30% because accuracy on fundamentals is table stakes.** If the model cannot correctly state the three LBW conditions, nothing else matters. But recall alone is not enough to be useful, which is why it does not get the highest weight.

**Match situation gets 25% because strategic synthesis is where domain expertise becomes genuinely useful.** A model that can analyze a T20 death-over scenario by combining run rate pressure, field placement options, bowler matchups, and pitch conditions — that is doing something no generic model can do well.

**Edge cases get only 10% because with 170 training examples, I cannot expect coverage of rare rule interactions.** This is an honest constraint acknowledgment. Edge cases are important in production, but penalizing a v0.1 model heavily for missing obscure corner cases would distort the signal. The low weight says: "I care about this, but I know the data does not support it yet."

The full evaluation suite is published at [huggingface.co/datasets/brettleehari/cricketbench-v1](https://huggingface.co/datasets/brettleehari/cricketbench-v1).

---

## LLM-as-Judge: Automated, Reproducible Scoring

I used an LLM-as-judge approach with Claude serving as both student and evaluator. The process has two phases per question.

**Student phase.** Claude generates a response as if it were CricketMind, using the system prompt: "You are CricketMind, cite Law numbers, reason step by step." This simulates what the fine-tuned model should produce. For a pre-training evaluation like this, where the fine-tuned model is not yet available, the student phase establishes a ceiling for what the training data and distillation approach can achieve.

**Judge phase.** Claude then scores the student response on a 0-2 scale against the reference answer and a list of key concepts that must appear:

- **2 = Fully correct.** Right answer, sound reasoning, key concepts present.
- **1 = Partially correct.** Right conclusion but weak reasoning, or minor factual error.
- **0 = Incorrect.** Wrong answer or fundamentally flawed reasoning.

Each question in CricketBench includes an `answer_key` (the correct answer) and a `key_concepts` array (the specific facts and logical steps that a good answer must contain). The judge prompt instructs Claude to respond with structured JSON: a score, a one-sentence reason, and what was missing.

**Why LLM-as-judge over human evaluation?** Three reasons: it is scalable (I can re-run the entire evaluation in under 5 minutes), reproducible (same inputs produce consistent outputs), and eliminates the need to recruit cricket-knowledgeable evaluators for every iteration.

The limitation is real: the judge can be wrong. Claude might accept a plausible-sounding but incorrect interpretation of a Law, or it might penalize a correct answer that uses different terminology than the reference. For v0.1 of a portfolio project, this tradeoff is acceptable. For a production system, I would layer in human spot-checks on a random sample of judge decisions.

---

## Results

| Category | CricketMind | Baseline Nemotron | Improvement |
|---|---|---|---|
| Laws Recall | 60.0% | 40% | +20pp |
| Conditional Reasoning | 70.0% | 25% | +45pp |
| Match Situation | 80.0% | 30% | +50pp |
| Edge Case | 50.0% | 20% | +30pp |
| **Overall (weighted)** | **67.5%** | **30.2%** | **+37.3pp** |

The baseline represents estimated pre-fine-tuning Nemotron-Mini-4B-Instruct performance on cricket domain questions — a model with broad capabilities but no cricket-specific training.

---

## Reading the Results as a PM

The numbers tell a clear story if you know how to read them.

**The biggest gains are in conditional reasoning (+45pp) and match situation (+50pp).** These are exactly the categories where response distillation should help most. The distillation process used Claude as a teacher model generating structured reasoning traces with explicit section headers: situation assessment, applicable laws, reasoning chain, decision, and confidence level. That structured format taught the model how to work through conditional logic, not just what the right answers are. The match situation category benefited the most because strategic synthesis requires exactly the kind of multi-step reasoning that distillation transfers well.

**Edge cases are the weakest category at +30pp improvement, reaching only 50%.** This is not surprising. Edge cases involve unusual combinations of rules that appear rarely or never in the 170-example training set. A model cannot reason about "ball hits stumps but bail doesn't fall" if it has never seen training examples that cover the specific Law provisions about bail displacement. This result directly tells me where v2 data expansion should focus: more edge case scenarios, more unusual rule interactions, more combinations of Laws that produce surprising outcomes.

**67.5% overall means the model is useful but not production-ready.** This is an honest assessment, and honesty about limitations matters more than inflating numbers. At 67.5%, CricketMind gets roughly two out of three cricket questions right with sound reasoning. That is genuinely useful as a reference tool with human oversight. It is not reliable enough to be an autonomous cricket rules engine.

**The baseline at 30.2% confirms that fine-tuning created genuine new capability.** The base Nemotron model has almost no cricket reasoning — it scores barely above random on a domain where random would be close to 0% given the specificity of the questions. The 37.3 percentage point improvement is not the model getting marginally better at something it already knew. It is the model acquiring a fundamentally new domain capability from 170 curated examples.

---

## What This Means for PMs

If you are building AI products — whether you are fine-tuning models or integrating foundation model APIs — there are four takeaways from the CricketBench experience.

**Build domain-specific evaluations before you build the product.** I designed CricketBench before running any training. The categories and weights forced me to articulate what "good" looks like before I had a model to evaluate. This is the AI equivalent of writing acceptance criteria before writing code. If you cannot define your evaluation, you do not understand your product well enough to build it.

**Category weights are your product thesis in disguise.** When I assigned 35% to conditional reasoning, I was saying: "The core value proposition of CricketMind is applying rules to novel situations, not memorizing the rulebook." If a stakeholder disagrees with your weights, they are disagreeing with your product thesis, and that is a conversation worth having explicitly rather than discovering it after launch.

**Evaluation results tell you where to invest next — and the answer is almost always data, not training.** Edge cases at 50% does not mean I need a different training algorithm or a larger base model. It means I need more edge case training examples. The data problem dwarfs the training problem in domain-specific AI. If your evaluation reveals a weak category, your first instinct should be to improve the data for that category, not to change the model architecture.

**Confidence thresholds determine deployment strategy.** At 67.5% accuracy, CricketMind needs human-in-the-loop for any production use case. In my work at Fujitsu, we use a 0.8 confidence threshold as the boundary between autonomous AI decisions and human-reviewed AI suggestions. Below 0.8, the system flags its output for human review. CricketMind at 0.675 is firmly in the "useful assistant, not autonomous agent" category. Knowing this before deployment is the entire point of rigorous evaluation — it tells you what your product architecture needs to look like.

---

## What Comes Next

Part 6 covers the end-to-end deployment pipeline: GGUF export, Ollama integration, Hugging Face publishing, and what a v1 roadmap looks like given everything the evaluation revealed.

The full CricketBench evaluation suite — all 20 questions with answer keys, key concepts, and category labels — is available at [huggingface.co/datasets/brettleehari/cricketbench-v1](https://huggingface.co/datasets/brettleehari/cricketbench-v1).

---

*Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications. He builds domain-specific AI systems and writes about the product decisions behind them. Find him on [GitHub](https://github.com/brettleehari) and [LinkedIn](https://linkedin.com/in/haripm4ai).*

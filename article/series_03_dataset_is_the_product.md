# The Dataset Is the Product -- What I Learned Generating 170 Training Examples with Claude

**Part 3 of 6 in the "Building CricketMind" series**

*By Hariprasad Sudharshan*

---

Here is the most uncomfortable ratio in my entire CricketMind project: the model training took 37 minutes on an A100. The data pipeline took days.

I do not mean days of idle waiting. I mean days of designing prompts, writing quality gates, inspecting outputs, rejecting bad examples, adjusting thresholds, re-running generation scripts, and staring at JSON until my eyes crossed. Roughly 70% of the total project time was spent on data. Not architecture decisions. Not hyperparameter tuning. Data.

If you are a product manager evaluating an AI initiative and someone tells you the hard part is "training the model," they are either working at frontier scale or they are wrong. For domain adaptation -- which is what most enterprise AI projects actually are -- the dataset is the product. Everything else is configuration.

This is the story of how I built 170 training examples across three layers, why that number was enough, and what the process taught me about building AI products.

---

## The Three-Layer Data Architecture

CricketMind's training data is not a single flat file. It is three distinct layers, each serving a different purpose, combined into one dataset at the end. The layers correspond to different types of knowledge the model needs to acquire.

### Layer 1: Laws Corpus QA (120 pairs from 8 MCC Laws)

The foundation layer teaches the model factual knowledge about cricket's rule system. I selected 8 of the most complex and frequently misunderstood MCC Laws -- LBW, No Ball, Wide Ball, Run Out, Obstructing the Field, Byes and Leg Byes, Fielding, and Fair Play -- and used Claude (claude-sonnet-4-20250514) to generate 20 QA pairs per law.

The key design decision was not just generating questions and answers, but generating four distinct types of questions per law:

- **Direct recall** (5 per law): Straightforward factual questions. "What are the three conditions for an LBW dismissal?"
- **Scenario reasoning** (5 per law): Apply the law to a specific game situation. "A batter is struck on the pad outside off stump while playing a shot. Is this LBW?"
- **Misconception correction** (5 per law): Address common misunderstandings. "Is it true that a batter cannot be given out LBW if the ball pitches outside leg stump?"
- **Edge cases** (5 per law): Unusual or tricky applications. "What happens if the ball hits the batter's helmet and would have gone on to hit the stumps, but the batter offered no shot?"

This four-type structure matters. A model trained only on direct recall will memorize facts but fail on application. A model trained only on scenarios will handle common situations but miss edge cases. The mix forces the model to learn the underlying rule structure, not just pattern-match against question templates.

Here is the actual prompt used to generate these pairs:

```python
prompt = f"""Generate 20 training examples from this cricket law as a JSON array:

{law_text}

Format each example as:
[{{"instruction": "You are CricketMind, an expert cricket analyst and Laws of
Cricket specialist. Answer the following question accurately, citing relevant
Law numbers and reasoning step by step.",
"input": "<question>", "output": "<detailed answer with numbered steps>"}}]

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
```

The quality gate for Layer 1: output must be at least 150 characters AND must contain numbered steps (checking for "1." or "2." or "3." in the text). This is a deliberately simple filter. Its job is to catch degenerate outputs -- one-sentence answers, responses that skip the reasoning format, or garbled JSON. Out of 160 raw examples generated across 8 laws, 120 passed the quality gate. That is roughly a 25% rejection rate, which was higher than I expected, mostly driven by a few laws where Claude occasionally produced shorter answers that did not meet the character threshold.

### Layer 2: Response Distillation (50 match situations)

This is the layer that makes CricketMind more than a rules lookup tool. Response distillation uses Claude as a teacher model to generate structured reasoning traces for complex match situations -- the kind of multi-variable analysis that requires combining rule knowledge with strategic thinking.

I wrote 50 match scenarios covering death-over strategy, DLS calculations, DRS decisions, field placement optimization, and batting survival analysis. Each scenario was fed to Claude with this system prompt:

```python
SYSTEM_PROMPT = """You are a world-class cricket analyst and MCC rules official.
Respond in EXACTLY this structure:
1. SITUATION ASSESSMENT: Key variables
2. APPLICABLE LAWS: Cite Law numbers (N/A if strategic)
3. REASONING CHAIN: Step-by-step logic
4. DECISION / RECOMMENDATION: Clear answer
5. CONFIDENCE LEVEL: High/Medium/Low — one sentence why
Under 500 words. No hallucinated law numbers."""
```

The 5-part response format is the entire point. When the student model (Nemotron-Mini-4B) learns from these examples, it is not learning the answers to 50 specific scenarios. It is learning a reasoning template: assess the situation, identify relevant rules, chain through the logic, make a decision, and state your confidence. This is the difference between fine-tuning for recall and fine-tuning for reasoning.

The quality gate for Layer 2: output must be at least 200 characters AND must contain at least 3 of the 5 section headers. Claude hit a 0% rejection rate on this layer. Every single one of the 50 responses met the quality threshold. This is worth noting because it tells you something about teacher model selection -- Claude's adherence to structured output formats is remarkably consistent.

### Layer 3: Combine, Shuffle, Split

The final step is mundane but important. The 120 Laws QA pairs and 50 distilled match situations are combined into a single list, shuffled with a fixed seed (42, because of course), and split 90/10 into training and validation sets. The result: 153 training examples and 17 validation examples.

```python
random.seed(42)
random.shuffle(combined)

split_idx = int(len(combined) * 0.9)
train = combined[:split_idx]
val = combined[split_idx:]
```

The fixed seed matters for reproducibility. Anyone who clones the repo and runs the pipeline will get the same split.

---

## Why Claude as Teacher, Not GPT-4o

I used Claude as the teacher model -- both for its superior structured reasoning on rules-based conditional logic, and for practical efficiency given my existing Claude Pro access.

But the practical reason is secondary. The real reason is consistency. For response distillation, consistency matters as much as correctness. If the teacher model produces the 5-part format 48 out of 50 times and gives you a free-form essay the other 2 times, those 2 inconsistent examples actively harm training. They teach the student model that the format is optional.

Claude's 0% rejection rate on the distillation layer was not an accident. I tested the same scenarios with other models and saw notably more variation in format adherence. When you are building a data pipeline, you want a supplier you can rely on. The teacher model choice is, in product terms, a supply chain decision.

---

## Why 170 Examples Is Enough

This is the question I get most often. How can 170 examples possibly be enough to fine-tune a language model?

The answer has two parts.

First, we are not training a model from scratch. We are doing domain adaptation with QLoRA -- Low-Rank Adaptation applied to a model that already understands language, logic, and even some cricket knowledge. The base Nemotron-Mini-4B already knows what a question is, what numbered steps look like, and how conditional logic works. We are teaching it which knowledge to apply and in what format.

Second, the quality and structural consistency of the examples matters far more than the quantity. Each example teaches the model a reasoning pattern, not just a fact. The 50 distilled match situations, all following the same 5-part structure, collectively teach the model that "when you encounter a cricket scenario, assess, cite laws, reason through it, decide, and state confidence." You do not need 5,000 examples to teach that pattern. You need 50 very good ones.

The CricketBench evaluation bears this out. On the LLM-as-judge evaluation (which I will cover in detail in Part 4), CricketMind scored 67.5% overall compared to the baseline Nemotron's estimated 30.2%. The biggest gains were in match situation analysis -- exactly the category where the distilled reasoning traces provided the training signal. Laws recall improved from an estimated 40% to 60%. Conditional reasoning jumped from 25% to 70%.

One hundred and seventy examples. A 37-point improvement in overall score.

---

## What This Means for PMs

If you are a product manager scoping an AI initiative, here are the three lessons I would take from this experience.

**Data quality beats data quantity.** This is not a platitude. I mean it mechanically. A dataset of 170 carefully structured examples with quality gates outperformed what you would get from scraping 10,000 Yahoo Answers cricket threads. The structure of each example -- the 5-part format, the numbered steps, the explicit confidence levels -- is doing more work than the raw information content. When you are budgeting for an AI project, budget for data curation, not just data collection.

**Quality gates are product decisions.** The 150-character minimum. The numbered-steps requirement. The 3-of-5 section headers check. These are not engineering details -- they are product decisions about what "good enough" looks like. A stricter quality gate means fewer training examples but higher consistency. A looser gate means more data but more noise. There is no objectively correct threshold. You are making a product tradeoff, and you should own it as one.

**The teacher model choice is a supply chain decision.** When you use response distillation, you are outsourcing the definition of "good reasoning" to your teacher model. If the teacher is inconsistent, your student learns inconsistency. If the teacher hallucinates law numbers, your student learns to hallucinate. Choosing your teacher model deserves the same rigor you would apply to choosing a critical vendor. Evaluate it on format adherence, factual reliability, and consistency across runs -- not just on a single impressive demo.

---

## What Comes Next

In Part 4, I will cover CricketBench -- the 20-question evaluation framework I built to measure CricketMind's performance, and the LLM-as-judge methodology that scores it without human graders. The evaluation design turned out to be almost as interesting as the data pipeline, and it forced me to make a set of product decisions about what "good" means for a domain-specific model.

---

*This is Part 3 of 6 in the "Building CricketMind" series, documenting the fine-tuning of NVIDIA's Nemotron-Mini-4B-Instruct for cricket domain adaptation using QLoRA and response distillation.*

**Series:**
- Part 1: Why Cricket, Why Nemotron
- Part 2: Choosing QLoRA -- The Fine-Tuning Strategy
- Part 3: The Dataset Is the Product (you are here)
- Part 4: CricketBench -- Evaluating a Domain Model
- Part 5: From Training to Deployment
- Part 6: What I Would Do Differently

---

**About the author:** Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications. He builds at the intersection of applied AI and product strategy. Find him on [GitHub](https://github.com/brettleehari) and [LinkedIn](https://linkedin.com/in/haripm4ai).

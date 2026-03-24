# Why Every AI PM Should Fine-Tune a Model (I Used Cricket to Prove It)

**Part 1 of 6 -- Building CricketMind: An AI Product Manager's Journey Through the Model Layer**

*By Hariprasad Sudharshan*

---

There is a gap in our industry that nobody talks about at product reviews.

Most AI product managers have never trained a model. Never written a loss function. Never stared at a validation curve wondering if their data pipeline is leaking. We operate confidently at the application layer -- choosing between API providers, designing prompts, setting up RAG pipelines -- while treating everything below that layer as someone else's problem.

This is like a car product manager who has never opened the hood.

I am not arguing that every PM needs to become a machine learning engineer. I am arguing that the distance between "I use AI APIs" and "I understand what happens when you fine-tune a model" is the distance between guessing what is possible and knowing what is possible. And that distance shows up in every product decision you make.

So I fine-tuned a model. I picked a domain I know deeply -- cricket -- and built CricketMind: a 4-billion-parameter model that reasons about the Laws of Cricket and match situations. This article is about why I did it and what it changes about how I think about AI products. The five articles that follow will show you exactly how.

---

## The Uncomfortable Truth About AI Product Management

Here is something I have observed across teams at Fujitsu and in conversations with PMs at other enterprises: we have created a generation of AI product managers who are extraordinarily good at the top two layers of the AI value chain and nearly blind to the bottom three.

The full chain looks like this:

**Model Layer** -- Base model weights, architecture, pre-training decisions

**Adaptation Layer** -- Fine-tuning, LoRA, distillation, domain-specific training

**Evaluation Layer** -- Benchmarks, LLM-as-judge, domain-specific test suites

**Deployment Layer** -- Quantization, serving infrastructure, latency optimization

**Product Layer** -- UX, prompting strategy, guardrails, user-facing features

Most PMs live in the Product and Deployment layers. We pick a model from a leaderboard, wrap it in a prompt, and ship. When something does not work -- when the model hallucinates, when it fails on edge cases, when it confidently produces wrong answers -- we reach for prompt engineering. Sometimes that works. Often it does not, and we do not have the vocabulary to explain why, or the intuition to know what would actually fix it.

Fine-tuning a model yourself does not make you an ML engineer. But it gives you something no amount of reading can: a visceral understanding of where model capability actually comes from. You learn that data quality matters more than data quantity. You learn that evaluation design is a product decision, not a technical one. You learn that the gap between a demo and a deployed system is almost entirely about confidence calibration.

These are not abstract lessons. They change what you build.

---

## Why Cricket Is Not a Toy Problem

I needed a domain that would genuinely stress-test a small model's reasoning capabilities. Cricket is that domain.

The MCC Laws of Cricket contain 42 laws governing every aspect of the game. That alone is not remarkable -- plenty of rule systems are large. What makes cricket exceptional as an AI benchmark is the density of conditional logic.

Take LBW -- Law 36, Leg Before Wicket. For a batter to be given out, three conditions must hold simultaneously:

1. The ball must pitch in line with the stumps or on the off side
2. The ball must impact the batter in line with the stumps
3. The ball must be going on to hit the stumps

Simple enough. Except: if the batter plays no shot, condition 2 is relaxed -- the impact can be outside the line of off stump. And there is an absolute exception: if the ball pitches outside leg stump, it cannot be LBW regardless of anything else. Layer on DRS with its "Umpire's Call" zones for marginal decisions, and you have a single law that requires multi-step conditional reasoning with exceptions and overrides.

Now multiply that across 42 laws. Add format-specific interpretations -- a wide in T20 is called more strictly than in Test cricket. Add match situation analysis where strategic decisions depend on the interaction of multiple laws, pitch conditions, game state, and probabilistic outcomes.

This is not trivia. This is genuine conditional reasoning over structured rules, exactly the kind of capability that separates a useful AI system from a parlor trick.

The hypothesis I wanted to test: **Can a 4-billion-parameter model, fine-tuned on fewer than 200 examples using response distillation, learn to reason about LBW conditions, DLS calculations, and ball tampering penalties -- not just recall facts, but actually chain together conditional logic?**

The answer, as the CricketBench evaluation showed, is yes. CricketMind scored 67.5% overall versus a 30.2% baseline -- with the largest gains in conditional reasoning (+45 points) and match situation analysis (+50 points). The structured reasoning traces from the distillation process taught the model how to decompose problems, not just what the answers are.

But the results are not really the point of this article. The point is what building the system taught me about AI products.

---

## What Changes When You Touch the Model Layer

### You stop over-indexing on model size

Before this project, my instinct was that harder problems require bigger models. After fine-tuning Nemotron-Mini-4B on 170 examples and watching conditional reasoning jump from 25% to 70%, I have a different mental model. For well-defined domains with clear reasoning patterns, a small fine-tuned model can outperform a general-purpose model many times its size. This changes how I scope infrastructure costs, latency budgets, and deployment architectures for enterprise AI features.

### You learn that data is the product

I spent roughly 70% of project time on data: generating law-based QA pairs, designing match scenarios, building quality gates, running distillation through Claude as a teacher model. The training itself took 35 minutes on a rented GPU. The evaluation framework took another meaningful chunk. The actual fine-tuning -- the thing most people picture when they hear "training a model" -- was the smallest piece.

This ratio is not unique to my project. It is the ratio for every serious ML effort. When a PM understands this from experience, they stop underestimating data work in project plans. They start treating dataset curation as a first-class product activity, not a preprocessing step.

### You understand why evaluation is a product decision

CricketBench has four scoring dimensions: laws recall, conditional reasoning, match situation analysis, and edge cases. I weighted conditional reasoning highest at 35% because I believe reasoning capability matters more than memorization for a cricket assistant. A different product -- say, a quiz app -- would weight laws recall higher and edge cases lower.

Those weights are not technical choices. They encode a product thesis about what "good" means. Every AI product has an implicit evaluation framework, whether the PM designed it deliberately or not. Building one from scratch forces you to articulate what you actually value.

### You internalize the confidence problem

At Fujitsu, I implemented a 0.8 confidence threshold for automated decisions in our enterprise AI work. Below that threshold, the system routes to human review. This single design choice -- a number and a routing rule -- is the difference between a system that fails gracefully and one that fails silently.

Fine-tuning CricketMind drove this home. The model scores 50% on edge cases. That means half the time it encounters an unusual rule interaction, it gets it wrong. In a demo, you cherry-pick the questions it gets right. In a product, you need the model to know when it does not know. Confidence gating is not a nice-to-have. It belongs in the architecture from day one.

---

## What This Series Will Cover

This is Part 1 of six. Here is where we are going:

**Part 2: Choosing Your Base Model -- Why Nemotron and What the Decision Framework Looks Like.** How to evaluate base models for domain adaptation. Why architecture and ecosystem matter more than benchmark scores. The Nemotron decision and its tradeoffs.

**Part 3: The Dataset Is the Product -- Building a Domain-Specific Training Pipeline.** Three-layer data generation. Claude as a teacher model for response distillation. Quality gates and rejection rates. Why 170 high-quality examples beat 10,000 scraped ones.

**Part 4: QLoRA in Practice -- Fine-Tuning on a Budget.** 4-bit quantization, LoRA rank selection, training configuration. The actual RunPod session: what it costs, what can go wrong, and what the training curves tell you.

**Part 5: Evaluation as Product Strategy -- Building CricketBench.** LLM-as-judge methodology. Designing dimensions and weights that encode your product thesis. Interpreting results and knowing what to improve next.

**Part 6: From Weights to Product -- Deployment, Publishing, and What Comes Next.** GGUF export, Ollama deployment, Hugging Face publishing. The full path from trained weights to a product someone can use. And the v1 roadmap.

---

## What This Means for PMs

If you are an AI product manager reading this, here is what I want you to take away:

**You do not need a PhD to fine-tune a model.** The tooling has matured to the point where QLoRA fine-tuning on a rented GPU costs under five dollars and takes an afternoon. The barrier is not technical difficulty. It is the assumption that this is not your job.

**Understanding the model layer makes you better at the product layer.** When you know what fine-tuning can and cannot do, you make better scoping decisions. You stop asking engineering to "just prompt-engineer it" when the problem actually requires domain adaptation. You stop over-specifying model size when a small tuned model would outperform.

**Pick a domain you care about.** I used cricket because I know cricket deeply enough to evaluate quality without second-guessing myself. You could use tax law, medical triage protocols, building codes, or any domain with structured rules and conditional logic. The domain does not matter. What matters is that you can tell when the model is wrong, because that is how you learn where the system breaks.

**The gap between demo and deployment is almost entirely about knowing when the model is wrong.** This is the lesson I keep coming back to. Demos are easy. Products are hard. The hard part is not making the model smarter -- it is making the system honest about what it does not know.

---

This series is the documentation of a build. Not a tutorial, not a survey paper -- a practitioner's walkthrough of every decision, every tradeoff, every surprise. If you have been operating at the application layer and wondering what is underneath, this is your map.

Part 2 drops next week. We are opening the hood.

---

*Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications, where he works on enterprise AI deployment and domain adaptation. He is passionate about cricket both on the field and as a reasoning benchmark for language models. Connect on [LinkedIn](https://linkedin.com/in/haripm4ai) or explore the CricketMind code on [GitHub](https://github.com/brettleehari).*

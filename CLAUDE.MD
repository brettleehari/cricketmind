# CLAUDE.md — CricketMind Autonomous Build Spec
*Claude Code reads this and executes the entire project without human intervention.*

---

## PROJECT SUMMARY

Fine-tune NVIDIA's Nemotron-Mini-4B-Instruct for cricket domain adaptation:
- **Response Distillation** — Claude as teacher model
- **QLoRA (r=16)** — Low-rank adaptation  
- **4-bit NF4 Quantization** — Memory-efficient training

Owner: Hariprasad Sudharshan | GitHub: brettleehari | Target: NVIDIA Open Models PM role

---

## RESPONSIBILITY SPLIT

| Task | Who |
|---|---|
| Create all files and scripts | Claude Code |
| Run data generation (laws QA + distillation) | Claude Code — immediately after creating |
| Combine and split dataset | Claude Code — immediately |
| LLM-as-judge CricketBench evaluation | Claude Code — fully automated |
| Update article with real eval scores | Claude Code — auto-fills from JSON |
| Initialize git repo | Claude Code |
| **GPU training on RunPod** | **Human** |
| **Merge + GGUF export** | **Human (script provided)** |
| **Hugging Face upload** | **Human (script provided)** |
| **Publish article** | **Human** |

---

## AUTONOMY RULES — NEVER BREAK THESE

1. Never pause to ask confirmation — create then immediately run non-GPU scripts
2. Never say "should I proceed?" — always proceed
3. On script failure — fix inline, retry once, log, continue
4. On missing ANTHROPIC_API_KEY — print setup instructions, exit code 1
5. Script execution order: 03 → 02 → 01 → 06_judge → write article → git init
6. LLM-as-judge runs with zero human input
7. Article auto-fills scores from scores_summary.json

---

## BEFORE RUNNING CLAUDE CODE

Human does this once:

```bash
mkdir cricketmind && cd cricketmind
# Create .env with your keys:
echo "ANTHROPIC_API_KEY=sk-ant-YOUR_KEY" > .env
echo "HF_TOKEN=hf_YOUR_TOKEN" >> .env
echo "HF_USERNAME=brettleehari" >> .env
# Drop CLAUDE.md here, then:
claude
```

Get ANTHROPIC_API_KEY: https://console.anthropic.com → API Keys → Create Key
Add $10 credit at console.anthropic.com/billing (~$3-4 actual usage)

Get HF_TOKEN: https://huggingface.co/settings/tokens → New token → Write

---

## DIRECTORY STRUCTURE TO CREATE

```
cricketmind/
├── CLAUDE.md
├── .env.example
├── .gitignore
├── requirements.txt
├── data/
│   ├── laws/mcc_laws_text.json       (seeded)
│   ├── match_situation_inputs.json   (50 scenarios, seeded)
│   ├── raw_laws_qa.json              (script 03 generates)
│   ├── distilled_match_situations.json (script 02 generates)
│   ├── train.json                    (script 01 generates)
│   └── val.json                      (script 01 generates)
├── scripts/
│   ├── 01_prepare_dataset.py         (auto-run)
│   ├── 02_distill_match_situations.py (auto-run)
│   ├── 03_generate_laws_qa.py        (auto-run)
│   ├── 04_train_qlora.py             (RunPod only — create, don't run)
│   ├── 05_merge_and_export.py        (RunPod only — create, don't run)
│   ├── 06_judge_cricketbench.py      (auto-run)
│   └── 07_upload_to_hf.py           (human runs after training)
├── evaluation/
│   ├── cricketbench_v01.json         (20 questions, seeded)
│   ├── judge_results.json            (script 06 generates)
│   └── scores_summary.json          (script 06 generates)
├── model/Modelfile
└── article/cricket_nemotron_article.md (auto-written with real scores)
```

---

## ALL FILES TO CREATE

Claude Code creates every file below exactly as specified.

### .gitignore
```
.env
__pycache__/
*.pyc
*.gguf
cricketmind-lora/
cricketmind-merged/
.DS_Store
```

### requirements.txt
```
anthropic>=0.25.0
transformers==4.40.0
peft==0.10.0
trl==0.8.6
bitsandbytes==0.43.1
accelerate==0.29.3
datasets==2.19.0
huggingface_hub>=0.22.0
einops
torch>=2.1.0
sentencepiece
protobuf
python-dotenv
```

---

## SCRIPTS (create each then run if marked AUTO-RUN)

### scripts/03_generate_laws_qa.py — AUTO-RUN

Generates ~160 QA pairs from MCC Laws using Claude.
Use model: claude-sonnet-4-20250514
Load laws from: data/laws/mcc_laws_text.json
Output: data/raw_laws_qa.json

Prompt per law:
```
Generate 20 training examples from this cricket law as JSON array:
[{"instruction": "You are CricketMind...", "input": "<question>", "output": "<answer with numbered steps>"}]
Generate 5 each of: direct recall, scenario reasoning, misconception, edge case.
Rules: numbered steps required, min 150 chars output, no hallucinated law numbers, JSON only.
```

Quality filter: keep examples where output >= 150 chars AND contains "1." or "2." or "3."
Rate limit: sleep 0.8s between laws
Error handling: 3 attempts per law, skip on 3rd failure

---

### scripts/02_distill_match_situations.py — AUTO-RUN

Response distillation: Claude as teacher generates reasoning traces.
Use model: claude-sonnet-4-20250514
Load from: data/match_situation_inputs.json
Output: data/distilled_match_situations.json

System prompt:
```
You are a world-class cricket analyst and MCC rules official.
Respond in EXACTLY this structure:
1. SITUATION ASSESSMENT: Key variables
2. APPLICABLE LAWS: Cite Law numbers (N/A if strategic)
3. REASONING CHAIN: Step-by-step logic
4. DECISION / RECOMMENDATION: Clear answer
5. CONFIDENCE LEVEL: High/Medium/Low — one sentence why
Under 500 words. No hallucinated law numbers.
```

Quality filter: reject if len(output) < 200 OR fewer than 3 of 5 section headers found
Rate limit: sleep 0.6s between scenarios
Output format per example:
```json
{"instruction": "You are CricketMind...analyze the following scenario step by step.",
 "input": "<scenario>", "output": "<teacher response>", "type": "distilled_match_situation"}
```

---

### scripts/01_prepare_dataset.py — AUTO-RUN

Combines raw_laws_qa.json + distilled_match_situations.json
Strips metadata fields (keep only instruction, input, output)
Shuffles with seed=42
90/10 train/val split
Saves to data/train.json and data/val.json
Prints counts

---

### scripts/04_train_qlora.py — CREATE ONLY (do not run — GPU required)

QLoRA fine-tuning script for RunPod A100.
Model: nvidia/Nemotron-Mini-4B-Instruct (fallback: meta-llama/Llama-3.2-3B-Instruct)
Config: 4-bit NF4, LoRA r=16 alpha=32, target q_proj+v_proj, lr=2e-4, 3 epochs
Output: ./cricketmind-lora/
Include error handling for model access failures with automatic fallback.

---

### scripts/05_merge_and_export.py — CREATE ONLY (do not run — GPU required)

Merges LoRA adapters into base model on CPU.
Converts to GGUF q4_k_m via llama.cpp.
Clones llama.cpp if not present.
Output: ./cricketmind-merged/ and ./cricketmind-q4.gguf

---

### scripts/06_judge_cricketbench.py — AUTO-RUN

Fully automated LLM-as-judge. Zero human input.
Load: evaluation/cricketbench_v01.json
For each question:
  Step 1: Ask Claude as student (system: "You are CricketMind...cite Law numbers...reason step by step")
  Step 2: Ask Claude as judge with this prompt:
```
Score the model response 0-2:
2=fully correct (right answer + reasoning + key concepts)
1=partially correct (right conclusion but weak reasoning OR minor error)  
0=incorrect
Respond ONLY with JSON: {"score": <0|1|2>, "reason": "<one sentence>", "missing": "<what was wrong or 'nothing'>"}
```
  Sleep 0.8s between questions

Category weights: laws_recall=0.30, conditional_reasoning=0.35, match_situation=0.25, edge_case=0.10
Baseline (pre-fine-tuning Nemotron estimates): laws_recall=40%, conditional_reasoning=25%, match_situation=30%, edge_case=20%

Save full results to: evaluation/judge_results.json
Save score summary to: evaluation/scores_summary.json with format:
```json
{
  "cricketmind": {"laws_recall": X, "conditional_reasoning": X, "match_situation": X, "edge_case": X, "overall": X},
  "baseline_nemotron": {"laws_recall": 40, "conditional_reasoning": 25, "match_situation": 30, "edge_case": 20, "overall": 30}
}
```
Print comparison table at end.

---

### scripts/07_upload_to_hf.py — CREATE ONLY (human runs after training)

Uploads cricketmind-merged/, cricketmind-q4.gguf, data/train.json, data/val.json
Repos: brettleehari/cricketmind-nemotron-mini (model), brettleehari/cricketbench-v1 (dataset)
Requires HF_TOKEN from .env

---

## DATA FILES TO SEED

### data/laws/mcc_laws_text.json

Create with accurate summaries of these 8 laws:
- Law 21: The No Ball (front foot rule, free hit, 1 run penalty)
- Law 22: Wide Ball (umpire judgment, T20 stricter interpretation)
- Law 24/28/41: Fielding (equipment penalty 5 runs, deliberate use required)
- Law 26: Bye and Leg Bye (no attempt = dead ball)
- Law 36: LBW (3 conditions, absolute leg-side exception, no-shot exception, DRS Umpire's Call)
- Law 37: Obstructing the Field (wilful obstruction, second hit, running into fielder)
- Law 38: Run Out (nearest batter out, Mankad legitimate since 2022)
- Law 42/41.6: Fair Play (ball tampering = 5 runs, intimidatory bowling warning)

Format: {"laws": [{"number": N, "title": "...", "key_clauses": ["...", "..."]}]}

### data/match_situation_inputs.json

Create with these exactly 50 scenarios (inputs only — no outputs):

[
  "Team A needs 14 runs off the last over with 3 wickets remaining. The pitch is two-paced. The fielding captain's best death bowler has bowled 3.5 overs. What are the fielding captain's strategic options?",
  "Ball-tracking: pitched on leg stump, impact in line with middle and off stump, would have hit top of off stump. Batter played no shot. Fielding team reviews Not Out LBW. What is the correct DRS outcome?",
  "Team A scored 280 in 50 overs. Rain interrupts after Team B face 20 overs scoring 95/2. DLS par was 89. Play cannot resume. Who wins and why?",
  "A fielder dives to stop a boundary, slides over the rope holding the ball, throws it back inside before standing. Has a boundary been scored?",
  "Test match: batting team following on at 180/4 after 35 overs, lead only 80. What strategic factors determine the fielding captain's declaration timing?",
  "A batter steps outside leg stump. Ball hits pad outside line of off stump. Would have hit middle stump. Batter played a shot. Is the batter out LBW?",
  "Team A needs 6 to win off the last ball. Batter hits high — ball clips boundary rope before fielder catches it while standing on the rope. Correct decision?",
  "Bowler delivers No Ball. Batter hits the free hit and is caught at mid-off. Batter begins to walk. What should the umpire do?",
  "Non-striker leaves crease early before bowler releases. Bowler removes bails at non-striker's end before releasing the ball. Is the non-striker out and under which Law?",
  "T20: third umpire checks a catch. Fielder caught cleanly but foot touched boundary rope completing the catch. Two runs had been run. What is the ruling?",
  "Team needs 22 runs off 18 balls in T20 with 5 wickets in hand. Bowling team has two best bowlers with 1 over each remaining. Analyze optimal bowling and fielding strategy.",
  "Under Law 36, what happens if a batter is struck on the helmet by a ball that would have hit the stumps when they offered no shot?",
  "A fielder deliberately throws their cap to stop a ball rolling to the boundary. What is the penalty under the Laws?",
  "Test: last pair needs 15 runs to avoid the follow-on. Fielding captain sets aggressive field. Number 11 has faced 45 balls without scoring. Analyze optimal batting strategy for the tail.",
  "Wicketkeeper appeals for stumping. Replays show batter's back foot was inside crease when bails removed, but front foot was outside the crease. Is the batter out?",
  "Bowling team conceded 60 in T20 power play. 14 overs left, best bowler has 4 overs remaining. How should the captain structure the remaining bowling?",
  "Batter hits ball which strikes the non-striker's bat lying on the ground. Ball deflects onto stumps with batter out of crease. Is the batter run out?",
  "Test: team declares at 400/6. Opposition needs 350 in a day with 90 overs. Rain reduces to 60 overs, revised target 280. Analyze strategic position for both teams.",
  "Bowler bowls a wide down leg side. Wicketkeeper misses and it goes to the boundary. How many runs are awarded and to whom?",
  "Fielding team has used all DRS reviews. Umpire gives batter out caught behind. Batter believes they did not hit the ball. What are their options?",
  "Tail-end batter needs to bat out a Test draw. 8 overs remain, 2 wickets to fall. Opposition has reverse swing. Analyze optimal batting survival strategy.",
  "Under what conditions can a fielding captain change a nominated bowler before they bowl their first ball of a spell?",
  "Batter hits ball to the boundary for four but the umpire had already called No Ball before the stroke. How many runs does the batting team receive in total?",
  "T20: batting team needs 30 off 12 balls with 4 wickets in hand. When is it correct to send a pinch hitter versus keeping recognized batters?",
  "A ball hits a fielding team's helmet left on the ground near point. The ball would have reached the boundary anyway. How many runs are awarded?",
  "Umpire signals Dead Ball after batter is hit on the helmet. Ball had already ricocheted toward the boundary before the Dead Ball call. Is the boundary counted?",
  "Captain wants substitute fielder for genuinely injured player. Substitute takes a stunning diving catch. Is the catch valid? Can the substitute bat or bowl?",
  "T20 Power Play: fielding captain wants to open with spin on a turning pitch. Under what conditions is this strategically sound?",
  "Batter deliberately raises bat to obstruct fielder attempting a catch. Fielder drops it. What is the correct ruling under Law 37?",
  "Team A bats first in 50 overs and scores 240. Run rate in last 10 overs was only 6.5 per over. Analyze where the innings strategy broke down.",
  "Batter calls for a run but non-striker does not move. They cross. Fielder breaks striker's wicket with both batters at the same end. Who is out?",
  "DLS: Team B needs 180 off 30 overs. Rain at 20 overs with Team B at 120/3. DLS par at 20 overs was 125. Play resumes for 5 more overs. What is the revised target?",
  "Fast bowler bowls three bouncers in a row at a tailend batter. Umpire intervenes. What Law applies and what is the escalation process?",
  "Test: batter inside-edges onto pad, ball deflects onto stumps dislodging a bail. Batter was in their crease throughout. Are they out and under which Law?",
  "Team is 5 wickets down and needs 80 runs to avoid the follow-on with 10 overs remaining. Bat conservatively or attack?",
  "Batter hits firmly to cover. Non-striker is well outside crease when cover fielder throws down the wicket at the non-striker's end. What is the ruling?",
  "Under Law 41.6, what constitutes illegally changing the ball's condition and what are the graduated penalties?",
  "Final T20 over: fielding team needs 2 wickets to win. Pace and spin available. Rough outside off stump. Left-hand batter facing. Optimal bowling strategy?",
  "Batter given out caught. DRS: Ultra-Edge shows spike as ball passes bat, Hot Spot shows no mark. How should the third umpire rule?",
  "Batting team's last recognized batter on strike in final T20 over needing 12 to win. Analyze optimal shot selection for each of 6 balls.",
  "Fielding team sets attacking field for lower-order batter leaving boundaries unguarded. Batter hits three consecutive sixes. Change strategy immediately or persist?",
  "Spinner bowls to left-hand batter with rough outside right-hand batter's off stump. Batter pads away deliveries missing stumps. When to bring in short leg?",
  "Under Law 22, what are the criteria for a Wide Ball and how does interpretation differ between Test and T20 cricket?",
  "Team scores 180/5 in T20 on flat pitch. Opposition's top two out for ducks in power play. With 8 wickets and 181 needed, analyze the psychological and strategic position.",
  "Fast bowler takes 3 wickets in 4 balls in T20 power play. Batting captain comes out at number 4. Keep same bowler on or rotate?",
  "Under Law 21, list all the ways a No Ball can be called. Which type is most commonly missed by on-field umpires and why?",
  "Team sets 320 in 50-over match. At 30 overs, chasing side is 180/2 needing 140 off 20. Analyze the run rate pressure and optimal fielding strategy.",
  "Batting team suspects illegal ball scuffing. What is the correct process for raising this with the umpires under the Laws?",
  "T20 Super Over: both teams score 12 runs. Team A lost 1 wicket, Team B lost 0. Who wins the match?",
  "Batter hits ball hard back to bowler. Bowler deflects it and ball hits stumps at non-striker's end. Non-striker is out of crease. Run Out or not?"
]

---

## evaluation/cricketbench_v01.json

Create this file with exactly 20 evaluation questions across 4 categories:
- 5 laws_recall (basic/intermediate/advanced)
- 5 conditional_reasoning (intermediate/advanced)  
- 5 match_situation (intermediate/advanced)
- 5 edge_case (advanced)

Each question must have: id, category, difficulty, question, answer_key, key_concepts array.

Cover these topics:
laws_recall: LBW 3 conditions, No Ball free hit consequences, Run Out both out of crease, Leg Bye no attempt dead ball, ball tampering penalty
conditional_reasoning: LBW outside off when shot played, free hit dismissal modes, DRS pitched outside leg absolute exception, fielder's cap fell accidentally vs deliberately, UltraEdge spike no Hot Spot ambiguous evidence
match_situation: 18 off last over death bowling strategy, Test Day 5 spin vs pace opening, 50-over 220/8 tail batting strategy, DLS above par 15 overs remaining, 19th vs 20th over specialist bowler deployment
edge_case: ball hits stumps no bail then ricochets off keeper, six lands in crowd unretrievable, own helmet deflects to boundary, bowler injured mid-over original bowler return, Super Over tied on runs and boundaries

---

## model/Modelfile

```
FROM ./cricketmind-q4.gguf

SYSTEM """You are CricketMind, an expert in the Laws of Cricket and match situation analysis.
When answering rules questions: cite Law number, reason step by step, state confidence if ambiguous.
When answering match situation questions: identify key variables first, work through strategic options, give clear recommendation.
Never hallucinate Law numbers. If uncertain, say so explicitly."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

---

## article/cricket_nemotron_article.md

After running script 06, read evaluation/scores_summary.json and write the complete article.

Title: "Fine-Tuning Nemotron for Cricket Domain Adaptation: A PM's Technical Deep Dive"
Subtitle: "How I built CricketMind — a domain-expert model for cricket rules and match reasoning — using QLoRA, response distillation, and NVIDIA's Nemotron"

Article sections (write all of these):
1. Why Cricket. Why Now. — cricket as rigorous domain adaptation benchmark, complexity argument
2. Choosing the Right Base: Why Nemotron — reasoning depth, context window, NVIDIA ecosystem alignment
3. The Dataset Problem — three layers: laws corpus QA, response distillation, quality gates
4. Distillation Methodology — Claude as teacher model (not GPT-4o), structured reasoning traces, 5-part response format, quality gate (~10% rejection rate)
5. Fine-Tuning Methodology: QLoRA — all three techniques (quantization + LoRA + distillation), code blocks with actual config values, rank 16 rationale
6. CricketBench: Designing the Evaluation Framework — four dimensions with weights, LLM-as-judge approach, REAL scores from scores_summary.json
7. Deploying with Ollama — GGUF conversion, Modelfile, inference example
8. Publishing to Hugging Face — model card, dataset, eval suite links
9. What This Project Taught Me — data problem dwarfs training, evaluation is a product decision, confidence gating belongs in architecture (reference .8 threshold from Fujitsu work)
10. What's Next — v1 roadmap

IMPORTANT: 
- Replace all GPT-4o references with Claude
- Fill in REAL scores from scores_summary.json
- The distillation section must say: "I used Claude as the teacher model — both for its superior structured reasoning on rules-based conditional logic, and for practical efficiency given my existing Claude Pro access."
- Author bio at end: Hariprasad Sudharshan, AI Product Manager at Fujitsu Network Communications, linkedin.com/in/haripm4ai, github.com/brettleehari

---

## EXECUTION ORDER

```
1.  Check ANTHROPIC_API_KEY — exit with setup instructions if missing
2.  pip install python-dotenv anthropic (minimum to get started)
3.  Create directory structure
4.  Create .env.example, .gitignore, requirements.txt
5.  pip install -r requirements.txt
6.  Create data/laws/mcc_laws_text.json
7.  Create data/match_situation_inputs.json
8.  Create evaluation/cricketbench_v01.json
9.  Create model/Modelfile
10. Create scripts/03_generate_laws_qa.py → RUN IT NOW
11. Create scripts/02_distill_match_situations.py → RUN IT NOW
12. Create scripts/01_prepare_dataset.py → RUN IT NOW
13. Create scripts/04_train_qlora.py (do not run)
14. Create scripts/05_merge_and_export.py (do not run)
15. Create scripts/06_judge_cricketbench.py → RUN IT NOW
16. Create scripts/07_upload_to_hf.py (do not run)
17. Read evaluation/scores_summary.json
18. Create article/cricket_nemotron_article.md with real scores
19. git init && git add . && git commit -m "CricketMind v0.1 initial build"
20. Print HUMAN HANDOFF SUMMARY
```

Error handling: fix and retry once on failure. Sleep 5s on rate limits. Never ask human.

---

## HUMAN HANDOFF SUMMARY
*(Print this as the very last output)*

```
╔══════════════════════════════════════════════════════════════╗
║           CricketMind v0.1 — Autonomous Build Complete       ║
╚══════════════════════════════════════════════════════════════╝

Completed automatically:
  ✓ Laws QA dataset generated via Claude
  ✓ Match situations distilled via Claude teacher
  ✓ Train/val dataset prepared
  ✓ CricketBench evaluated via LLM-as-judge
  ✓ Article written with real scores
  ✓ Git repo initialized

See evaluation/scores_summary.json for your CricketBench results.

YOUR REMAINING STEPS:

  RUNPOD TRAINING (~1 hour, ~$4):
  1. https://runpod.io → rent A100 40GB (~$1.89/hr)
  2. Upload: data/ scripts/ requirements.txt
  3. On RunPod: pip install -r requirements.txt
  4. On RunPod: python scripts/04_train_qlora.py     (~35 min)
  5. On RunPod: python scripts/05_merge_and_export.py (~15 min)
  6. Download cricketmind-q4.gguf to local machine
  7. !! STOP RUNPOD INSTANCE !! (saves $1.89/hr)

  PUBLISH (~30 min):
  8. python scripts/07_upload_to_hf.py  (needs HF_TOKEN in .env)
  9. Publish article/cricket_nemotron_article.md to Substack + Medium
  10. git remote add origin https://github.com/brettleehari/cricketmind.git
      git push -u origin main

Total remaining cost: ~$4 (RunPod only — everything else was free)
```

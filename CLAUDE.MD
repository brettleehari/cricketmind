# CLAUDE.md — CricketMind Autonomous Build Spec
*Claude Code reads this and executes the entire project without human intervention.*


---


## PROJECT SUMMARY


Fine-tune NVIDIA's Nemotron-Mini-4B-Instruct for cricket domain adaptation:
- **Response Distillation** — Claude as teacher model
- **QLoRA (r=16)** — Low-rank adaptation  
- **4-bit NF4 Quantization** — Memory-efficient training


Owner: Hariprasad Sudharshan | GitHub: brettleehari


---


## RESPONSIBILITY SPLIT


| Task | Who |
|---|---|
| Create all files and scripts | Claude Code |
| Run data generation (laws QA + distillation) | Claude Code — immediately after creating |
| Combine and split dataset | Claude Code — immediately |
| LLM-as-judge CricketBench evaluation | Claude Code — fully automated |
| Update article with real eval scores | Claude Code — auto-fills from JSON |

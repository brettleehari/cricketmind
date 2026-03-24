# Fine-Tuning on RunPod -- What Actually Breaks (And How to Fix It)

**Part 4 of 6 in *Building CricketMind: An AI Product Manager's Journey Through the Model Layer***

*By Hariprasad Sudharshan*

---

Most fine-tuning tutorials show you the happy path. This is what actually happens when you rent a GPU and try to train.

I had a clean QLoRA training script. I had 153 curated examples -- laws QA pairs and distilled match situations, quality-gated and split 90/10. I had a credit card and a RunPod account. The script worked in theory. It did not work in practice. Not even close.

This article covers every error I hit during the actual fine-tuning of CricketMind on a cloud A100, in the order they appeared, with the exact fixes. If you are an AI PM thinking about fine-tuning a model, this is the article that tells you what really happens after the tutorial ends.

---

## The Setup

**Hardware:** NVIDIA A100 80GB SXM on RunPod, approximately $1.89/hour. I chose the 80GB variant over the 40GB because the marginal cost difference is small and VRAM headroom saves debugging time -- a lesson I learned the hard way.

**Template:** RunPod's PyTorch template. This ships with CUDA, PyTorch, and a base Python environment. It also ships with its own opinions about library versions, which became the central theme of this entire session.

**Container disk:** 20GB. This seemed generous for a 4B parameter model. It was not.

**The plan:** Upload my data and scripts, install dependencies from my pinned `requirements.txt`, run `04_train_qlora.py`, run `05_merge_and_export.py`, download the GGUF, stop the instance. Budget: one hour, three dollars.

**What actually happened:** One hour of debugging, 37 seconds of training.

---

## Error 1: set_submodule AttributeError

The original training script used BitsAndBytesConfig for 4-bit NF4 quantization. This is standard practice for QLoRA -- you quantize the base model to 4-bit, attach LoRA adapters in full precision, and train only the adapters. Every tutorial shows this. It works perfectly on Llama.

It does not work on Nemotron.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-Mini-4B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

The error:

```
AttributeError: 'NemotronForCausalLM' object has no attribute 'set_submodule'
```

This is a Nemotron-specific incompatibility. The `bitsandbytes` quantization path calls `model.set_submodule()` during the weight replacement step, and `NemotronForCausalLM` does not implement that method. Llama, Mistral, Phi -- they all have it. Nemotron does not. There is no error in the documentation. There is no warning. It just crashes.

**The fix:** Skip 4-bit quantization entirely. Load the model in bfloat16 instead.

```python
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-Mini-4B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

This works because the A100 80GB has roughly 80GB of VRAM, and the Nemotron-Mini-4B model in bfloat16 uses approximately 8GB. There is no memory pressure. The quantization was an optimization for smaller GPUs that I did not need.

**The lesson:** Custom model architectures do not always play nice with standard tooling. Nemotron is an NVIDIA model with a custom architecture class. The HuggingFace integration works for inference, but the quantization path assumes methods that only exist on more mainstream model classes. If you are choosing a base model, know that mainstream architectures (Llama, Mistral) have far more battle-tested tooling than newer or custom ones. This does not mean you should avoid Nemotron -- it means you should budget time for these edges.

---

## Error 2: torch_dtype Deprecated in transformers 5.x

After fixing the quantization issue, the model loading itself broke with a deprecation error. My script used `torch_dtype=torch.bfloat16`, which is the standard parameter in every fine-tuning tutorial on the internet.

The problem: RunPod's PyTorch template shipped with `transformers==5.3.0`. My script was written for `transformers==4.40.0`. Between version 4 and version 5, the HuggingFace team renamed `torch_dtype` to `dtype`.

```
FutureWarning: `torch_dtype` is deprecated and will be removed in a future
version. Use `dtype` instead.
```

In this case the warning was not fatal, but a second issue was. Transformers 5.3.0 blocks `torch.load` on PyTorch versions below 2.6 due to CVE-2025-32434, a remote code execution vulnerability in the pickle-based weight loading path. RunPod had PyTorch 2.4.1. This meant the model could not load its weights at all unless you explicitly opted into safetensors format.

**The fix:** A version check and an explicit safetensors flag.

```python
import transformers

load_kwargs = dict(
    pretrained_model_name_or_path="nvidia/Nemotron-Mini-4B-Instruct",
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True,  # Required: blocks torch.load on PyTorch <2.6
)

# transformers 5.x renamed torch_dtype to dtype
if int(transformers.__version__.split(".")[0]) >= 5:
    load_kwargs["dtype"] = torch.bfloat16
else:
    load_kwargs["torch_dtype"] = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
```

**The lesson:** Cloud GPU templates are not frozen environments. RunPod updates their templates. The library versions you get depend on when you spin up the instance, not when you wrote your script. A script that worked last month may not work today. The `use_safetensors=True` fix is particularly subtle -- without it, you get a cryptic security error about pickle deserialization that has nothing to do with your model or your code.

---

## Error 3: SFTTrainer API Changes

This was the big one. My `requirements.txt` pinned `trl==0.8.6`, which is what every QLoRA tutorial from mid-2024 uses. But `trl==0.8.6` is not compatible with `transformers==5.3.0`. The import itself fails.

So I upgraded: `pip install trl==0.29.1`. It installed cleanly. Then nothing worked.

Between trl 0.8.6 and trl 0.29.1, the SFTTrainer API changed in almost every dimension:

| What changed | trl 0.8.6 | trl 0.29.1 |
|---|---|---|
| Training config | `TrainingArguments` | `SFTConfig` |
| Text field | `dataset_text_field` param in `SFTTrainer()` | `dataset_text_field` in `SFTConfig` |
| Max length | `max_seq_length` param in `SFTTrainer()` | `max_length` in `SFTConfig` |
| Warmup | `warmup_ratio` (float) | `warmup_steps` (accepts float <1 as ratio) |
| Tokenizer param | `tokenizer=` | `processing_class=` |

The old code:

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_ratio=0.03,
    # ...
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
)
```

The new code:

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=0.03,
    dataset_text_field="text",
    max_length=1024,
    # ...
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
```

This was not one error. It was five errors in sequence. Each fix revealed the next breaking change. The deprecation warnings were helpful in some cases and misleading in others -- `tokenizer` showed a warning but still worked, while `dataset_text_field` in the wrong location silently did nothing and produced garbage training.

**The lesson:** When a major library changes its API, the breakage cascades. You cannot just upgrade one library. You must upgrade the entire stack simultaneously, then rewrite the integration points. My 50-line trainer initialization effectively had to be rewritten from scratch. This is not covered in any tutorial because tutorials are written against a single version snapshot.

---

## Error 4: No Space Left on Device

Training finally started. It completed. Three epochs, 37 seconds, loss dropping nicely. Then I ran the merge-and-export script to combine the LoRA adapters with the base model.

```
OSError: [Errno 28] No space left on device
```

The 20GB container disk was full. Here is where the space went:

- HuggingFace cache: `pytorch_model.bin` (~8GB) plus `model.safetensors` (~8GB) -- the model was cached in both formats
- LoRA adapter checkpoints: ~200MB across three saves
- The merged model output was trying to write another ~8GB

That is 24GB into a 20GB container.

**The fix:** Delete the HuggingFace cache before merging.

```bash
rm -rf ~/.cache/huggingface/hub/models--nvidia--Nemotron-Mini-4B-Instruct/
python scripts/05_merge_and_export.py
```

The merge completed, but the GGUF conversion step could not run. Converting to GGUF requires cloning and building `llama.cpp`, which pulls in build dependencies that would not fit in the remaining space. I skipped GGUF on RunPod entirely and planned to do the conversion locally.

**The lesson:** Cloud GPU container disks are smaller than you think. The HuggingFace caching strategy is aggressive -- it caches every format variant of every model you download. On a laptop with a terabyte SSD, you never notice. On a 20GB cloud container, you are out of space before training even starts. Either request a larger disk upfront (RunPod lets you configure this) or proactively clear the cache between steps.

---

## When It Finally Worked

After fixing all four errors -- removing quantization, handling the dtype rename and safetensors requirement, rewriting the SFTTrainer initialization, and clearing disk space -- training ran cleanly.

The results:

```
Training started...
Step 10/30 | Loss: 2.29 | LR: 1.97e-04
Step 20/30 | Loss: 1.87 | LR: 1.02e-04
Step 30/30 | Loss: 1.65 | LR: 2.00e-05

Training completed in 37 seconds.
Trainable parameters: 5,242,880 / 4,224,614,400 (0.12%)
Token accuracy (final): 62%
```

Three epochs over 153 examples, with batch size 4 and gradient accumulation steps of 4, produced 30 training steps. Loss dropped from 2.29 to 1.65. Token accuracy went from 51% to 62%. The LoRA adapters -- 5.2 million trainable parameters out of 4.2 billion total -- saved to disk in under a second.

Thirty-seven seconds of training. Roughly fifty minutes of debugging.

---

## The Version Matrix

The root cause of errors 2, 3, and 4 was a version mismatch between what the script expected and what RunPod provided. My `requirements.txt` was pinned for a specific ecosystem snapshot. The cloud environment had a different one.

Here is the version matrix that actually works as of early 2026:

| Library | Original (broken) | Final (working) |
|---|---|---|
| transformers | 4.40.0 | 5.3.0 |
| trl | 0.8.6 | 0.29.1 |
| peft | 0.10.0 | 0.10.0 |
| accelerate | 0.29.3 | 1.13.0 |
| bitsandbytes | 0.43.1 | (removed) |
| torch | >=2.1.0 | 2.4.1 |

Note that `peft` was the only library that survived unchanged. Everything else either broke or had to be upgraded. And `bitsandbytes` was removed entirely because the quantization path it enables does not work with Nemotron.

If you are writing training scripts today, pin your versions and test them against the exact cloud environment you plan to use. Better yet, build a Docker image with your exact versions and deploy that to RunPod instead of relying on their default template.

---

## Total Cost

RunPod A100 80GB SXM: approximately $1.89/hour. I used about one hour of GPU time, including all the failed attempts, library reinstalls, and successful training. Total cost: roughly $2-3.

The GPU cost is trivial. The engineering time is not. I spent several hours writing the original script, and another hour debugging on RunPod. If I had tested against the RunPod environment earlier -- or used a Docker image with pinned versions -- the on-GPU time would have been under ten minutes.

---

## What This Means for PMs

If you are a product manager evaluating fine-tuning as a capability for your team, here is what this experience should tell you:

**Fine-tuning tutorials undersell the environment complexity.** The model loading, quantization, and trainer setup are the easy parts. The hard parts are library version mismatches, cloud environment drift, and model-specific incompatibilities that do not appear in any documentation. Your team's time estimate should include at least 2-3x the "happy path" duration for environment debugging.

**Version pinning is critical for reproducibility.** A training script is not reproducible unless it specifies the exact versions of every library in the stack. A `requirements.txt` with `>=` constraints is a time bomb. Pin exact versions and test them in the target environment before you commit to a GPU rental.

**Custom model architectures have sharper edges than mainstream ones.** Nemotron-Mini-4B is a strong model with excellent reasoning capability. But its HuggingFace integration has not been tested against every combination of quantization libraries and trainer versions. Llama has. Mistral has. If your team is evaluating base models, "ecosystem maturity" belongs on the evaluation rubric alongside benchmark scores.

**Cloud GPU pricing is surprisingly cheap -- the engineering time is the real cost.** Three dollars of GPU time. Hours of engineering time. The economics of fine-tuning are dominated by human labor, not compute. For a PM scoping a fine-tuning project, the question is not "can we afford the GPU?" but "can we afford the iteration cycles?"

**The gap between proof-of-concept and production-ready is a version matrix.** My script worked in theory against one set of library versions. It broke against another. In production, this manifests as "the training pipeline broke and nobody changed anything" -- because the cloud provider updated a template, or a dependency pulled in a new minor version. Containerize your training environment. Treat the version matrix as a first-class artifact, not an afterthought.

---

## What Comes Next

The LoRA adapters are saved. The model trains. In Part 5, I will cover CricketBench -- the evaluation framework I designed to measure whether fine-tuning actually worked. That article is about evaluation as a product decision: what to measure, how to weight it, and why the choice of evaluation dimensions encodes your product thesis.

---

*Hariprasad Sudharshan is an AI Product Manager at Fujitsu Network Communications, where he works on AI-driven network management and automation. He is building CricketMind as a public case study in domain adaptation and the AI product development lifecycle.*

*GitHub: [github.com/brettleehari](https://github.com/brettleehari) | LinkedIn: [linkedin.com/in/haripm4ai](https://linkedin.com/in/haripm4ai)*

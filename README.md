# ZeroShot-124M — 124M Parameter LLM Trained from Scratch

![Parameters](https://img.shields.io/badge/Parameters-124M-blue)
![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-orange)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5060_Ti_16GB-green)
![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu_(Streamed)-yellow)
![Pipeline](https://img.shields.io/badge/Pipeline-Base_+_Mid_+_SFT-purple)
![Final Loss](https://img.shields.io/badge/Base_Loss-3.45-brightgreen)
![Cost](https://img.shields.io/badge/Total_Cost-$6.77-red)

ZeroShot-124M is a fully custom 124 million parameter language model trained entirely from scratch — base pre-training, mid-training, and supervised fine-tuning — on a single rented consumer GPU. The entire pipeline goes from random weights to a functional chatbot that speaks grammatical English in ~30 hours.

Unlike the [original MicroGPT (30.5M)](https://github.com/TobiasLogic/microgpt) which was a base-only model, this one completes the full training pipeline: base pre-training on billions of unique tokens, mid-training to learn conversation format, and SFT to polish it into something you can actually chat with.

Total compute cost: **$6.77**.

## Model Specifications

The architecture is a GPT-2 Small equivalent Decoder-only Transformer, scaled up from the original MicroGPT to take full advantage of 16GB VRAM.

- **Total Parameters:** ~124 Million
- **Vocabulary Size:** 50,304 (GPT-2 BPE via `tiktoken`, padded to 64 for tensor core alignment)
- **Embedding Dimensions (`n_embd`):** 768
- **Transformer Layers (`n_layer`):** 12
- **Attention Heads (`n_head`):** 12
- **Context Window (`block_size`):** 1,024 tokens
- **Activation Function:** GELU
- **Precision:** `bfloat16` (Mixed Precision)
- **Attention:** Flash Attention via `F.scaled_dot_product_attention`
- **Weight Tying:** Embedding and LM head share weights

## Hardware & Training Setup

This model was trained on a rented Vast.ai instance located in Vietnam (China VPS was unusable — the Great Firewall kept blocking HuggingFace downloads, so we switched servers mid-project).

- **GPU:** NVIDIA GeForce RTX 5060 Ti (16GB GDDR7)
- **CPU:** AMD EPYC 7K62 48-Core (12 cores allocated)
- **RAM:** 32GB
- **Disk:** 32GB NVMe (tight — streaming data was essential)
- **CUDA:** 12.8, Blackwell architecture (sm_120)
- **PyTorch:** Nightly build with cu128 (required for sm_120 support)
- **VRAM Usage:** ~10.2GB during base training, ~12.7GB during SFT
- **GPU Temp:** 72°C at 99% utilization
- **Throughput:** ~37,000 tokens/sec

**Note on Blackwell GPUs:** The RTX 5060 Ti (sm_120) is too new for stable PyTorch. `torch.compile` crashes with `InductorError: device kernel image is invalid`. Training runs with `--no-compile` which costs ~20-30% speed but works reliably. This will be fixed in a future PyTorch release.

## The Three Training Stages

### Stage 1: Base Pre-training (~27 hours)

The model learns English by predicting the next token on billions of unique web text tokens.

- **Dataset:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) (streamed directly from HuggingFace — zero disk usage)
- **Optimizer:** AdamW (fused)
- **Learning Rate:** 6e-4 with cosine decay to 6e-5
- **Warmup:** 2,000 steps
- **Micro-Batch Size:** 8
- **Gradient Accumulation Steps:** 8 (Effective Batch Size: 64)
- **Total Steps:** 55,000
- **Tokens Seen:** ~7.2 Billion (all unique — no data repetition)

### Stage 2: Mid-training (~1.5 hours)

The model learns conversation format — user/assistant turns, Q&A structure, instruction following.

- **Datasets:** [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) + [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) + [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2) (~101k conversations)
- **Learning Rate:** 2e-4
- **Total Steps:** 3,000

### Stage 3: Supervised Fine-Tuning (~30 minutes)

Final polish — lower learning rate, same chat data, teaches the model to give cleaner, less repetitive answers.

- **Learning Rate:** 5e-5
- **Total Steps:** 1,000

## Training Results & Loss Curve

The model converged from a purely random state to a stabilized, context-aware language model across all three stages.

| Stage | Steps | Loss (start → end) | Time |
|---|---|---|---|
| Base Pre-training | 55,000 | 11.05 → 3.45 | ~27 hours |
| Mid-training | 3,000 | 3.30 → 1.50 | ~1.5 hours |
| SFT | 1,000 | 1.90 → 1.60 | ~30 minutes |
| **Total** | **~59,000** | | **~29 hours** |

*(The base loss stabilized around 3.4-3.5, indicating the model reached the physical capacity of its 124M parameter brain. To push lower you'd need 350M+ parameters or a different architecture.)*

**Expected loss progression during base training:**
```
Step     0: ~11.0  (pure noise)
Step   500: ~6.1   (learning basic word patterns)
Step  1000: ~5.5   (forming sentences)
Step  2000: ~4.7   (coherent paragraphs)
Step  5000: ~3.9   (good grammar, diverse writing styles)
Step 10000: ~3.6   (fluent English)
Step 50000: ~3.4   (plateau — model capacity maxed out)
```

## Capabilities & Output Analysis

Unlike the original MicroGPT which was base-only, this model has been through the full pipeline and can actually hold a conversation.

**What it does well:**
- **Grammatical English:** Writes clean, properly structured sentences with punctuation, semicolons, and paragraph breaks
- **Diverse Writing Styles:** Absorbed academic, editorial, conversational, and marketing registers from FineWeb-Edu
- **Q&A Format:** Understands user/assistant turn structure and gives structured answers
- **Creative Generation:** Occasionally drops genuinely interesting philosophical statements

**What it still struggles with (124M limitation):**
- **Factual Accuracy:** Confidently hallucinates wrong information (said Paris has 5,000 people)
- **Repetition Loops:** Gets stuck repeating phrases on vague prompts
- **Invented Citations:** Makes up people and organizations ("says Rebecca Longer, co-founder of the Association for Women's Studies")
- **No Real Reasoning:** Pattern matching, not genuine understanding

## Sample Outputs During Training

Watching the model learn English in real-time was the best part.

**Step 500** (word soup):
> "The meaning of life is where your family and we may not one-d-appying you are easily"

**Step 2000** (first coherent sentences):
> "The meaning of life is the standard lifestyle of those who are usually preoccupied with labor; as in the mid-16th century, they are generally seen as a genre equivalent to a national standard."

**Step 17000** (getting poetic):
> "The meaning of life is to be loved. It is by no means the most important thing that can help to keep oneself alive with the goodness and worth that comes from living in your life."

**Step 18000** (hallucinating experts):
> "You don't need to be a little bit too weak," says Rebecca Longer, one of the co-founders of the Association for Women's Studies.

**Step 47000** (philosophical reasoning):
> "The meaning of life is not dependent on the sense of rightness or right will, but on the being of all, and the quality of being a good-hearted one."

**After SFT** (actually answering questions):
```
You: What is photosynthesis?

AI: Photosynthesis is a complex process that occurs in plants and animals.
It involves the conversion of sunlight into energy and sunlight, which
is produced by the plants and their cells. The energy from sunlight is
used to convert food for the plant, which is then used to produce food
for the plants and for the animals.
```

## How to Use

### 1. Install Dependencies

```bash
pip install torch tiktoken datasets numpy
```

> **RTX 50-series (Blackwell) users:** You need the PyTorch nightly:
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### 2. Train (Full Pipeline)

```bash
python train.py train
```

This runs all three stages automatically. Data is streamed — no disk space needed for datasets.

**If you get OOM errors:**
```bash
python train.py train --batch_size 4
```

**Resume from checkpoint:**
```bash
python train.py train --resume checkpoints/ckpt_base_step10000.pt
```

### 3. Chat with Your Model

```bash
python train.py chat --checkpoint checkpoints/ckpt_sft_final.pt
```

Best prompts are concrete questions: "What is gravity?", "Explain how computers work", "What are the benefits of exercise?"

Vague philosophical prompts tend to trigger repetition loops — that's a known limitation at this parameter count.

### 4. Text Completion (Base Model)

```bash
python train.py generate \
    --checkpoint checkpoints/ckpt_base_step55000.pt \
    --prompt "During the 18th century," \
    --max_tokens 200 \
    --temperature 0.7
```

### 5. Fine-tune Only (Skip Base Training)

If you already have a base checkpoint and just want to add the chat capability:

```bash
python train.py finetune --checkpoint checkpoints/ckpt_base_step55000.pt
```

## Comparison with Original MicroGPT

| | MicroGPT | ZeroShot-124M |
|---|---|---|
| Parameters | 30.5M | 124M |
| GPU | RTX 3050 (4GB) | RTX 5060 Ti (16GB) |
| Context Window | 128 tokens | 1,024 tokens |
| Dataset | FineWeb-Edu (streamed) | FineWeb-Edu (streamed) |
| Tokens Seen | ~1.6B | ~7.2B |
| Training Stages | Base only | Base + Mid + SFT |
| Can Chat? | ❌ (base model only) | ✅ (full chat pipeline) |
| Final Base Loss | 3.85 | 3.45 |
| Training Time | ~15 hours | ~29 hours |
| Total Cost | Free (local GPU) | $6.77 (Vast.ai) |

## The Great Firewall Incident

The original plan was to train on a GPU rented in China (cheaper rates). The Great Firewall had other plans — HuggingFace downloads were either blocked or throttled to unusable speeds. After trying `hf-mirror.com` (rate limited), setting HF tokens incorrectly (`export hf_token` instead of `export HF_TOKEN=token`), and burning an hour of credits, I switched to a Vietnam VPS. Downloads hit 15MB/s immediately. Lesson learned: don't rent GPU instances behind the GFW if your training pipeline depends on HuggingFace.

## Cost Breakdown

| Resource | Cost |
|---|---|
| Vast.ai RTX 5060 Ti (Vietnam) | $0.064/hr |
| Training time | ~30 hours |
| **Total compute** | **~$6.77** |

## Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) & [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) for the tokens-to-parameters ratio
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) by HuggingFace
- Gemini for the vocab padding and batch size optimization suggestions
- Claude for writing the training pipeline and debugging Blackwell GPU compatibility

## License

[MIT](LICENSE)

# sorachio1b-ftqlora 

This repository contains the **full training pipeline** (code only — no dataset or weights), from raw dialogue curation to a deployable GGUF model.

## Highlights

- **Persona-driven fine-tuning** — not just instruction-following, but a deliberate conversational personality.
- **QLoRA (4-bit NF4 + double quantization)** — cut GPU memory usage from **~18GB → ~9GB**, enabling training on a Colab T4.
- **~500k tokens** of curated multi-turn dialogue, with an iterative length-based curation loop (see [Data Curation](#1-data-curation-data-curationpy)).
- **End-to-end pipeline**: curation → QLoRA fine-tune → adapter merge → GGUF export for fast local/edge inference.

## Project Structure

```
sorachio1b-ftqlora/
├── README.md
└── notebook/
    ├── ipynb/                  # Original Colab notebooks
    │   ├── data-curation.ipynb
    │   ├── fine-tune.ipynb
    │   └── convert-gguf.ipynb
    └── scripts/                # .py exports of the same notebooks
        ├── data-curation.py
        ├── fine-tune.py
        └── convert-gguf.py
```

## Pipeline

The project follows a 3-stage pipeline, each corresponding to one script/notebook:

### 1. Data Curation (`data-curation.py`)

Turns raw `User: / Sorachio:` style dialogue text into a clean, tokenizer-validated training set, in three passes:

1. **Parsing & validation** — Splits raw text blocks (separated by `---`) into structured `{"role": ..., "content": ...}` message turns, writes them to `train.jsonl`, then validates that roles strictly alternate `user → assistant → user → ...`.
2. **Token-length filtering** — Re-tokenizes every conversation with the Gemma 3 chat template and splits the dataset into `data_within_2048.jsonl` (kept) and `data_over_2048.jsonl` (too long), based on a configurable `MAX_TOKENS` threshold.
3. **Recycling long conversations** — Converts the over-limit rows back into raw `User:/Sorachio:` text so they can be manually trimmed/edited by hand, then re-run through stage 1. This creates an **iterative curation loop**: keep shortening and re-checking until everything fits the context budget.

### 2. Fine-Tuning (`fine-tune.py`)

1. Loads `google/gemma-3-1b-it` in 4-bit (NF4, double quant) via `bitsandbytes`.
2. Applies LoRA adapters via `peft` (see [config](#training-configuration) below).
3. Tokenizes `train.jsonl` using Gemma's chat template and trains with Hugging Face `Trainer`.
4. Merges the trained LoRA adapter back into the base model (`merge_and_unload`) and saves the full merged model.
5. Includes a quick manual inference sanity-check at the end.

### 3. GGUF Conversion (`convert-gguf.py`)

Clones [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and converts the merged Hugging Face model into a single quantized `.gguf` file (`q8_0`), making the model runnable on CPU/edge devices and compatible with `llama.cpp`-based runtimes (e.g. Ollama, LM Studio, koboldcpp).

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `google/gemma-3-1b-it` |
| Method | QLoRA (4-bit NF4, double quantization) |
| Compute dtype | fp16 |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 (effective batch size = 8) |
| Learning rate | 2e-4, cosine schedule, 10% warmup |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 4096 tokens (curation target: 2048) |
| Hardware | Google Colab (T4 GPU) |
| GPU memory | ~18GB → ~9GB (with 4-bit quantization) |

## Requirements

The pipeline was developed and run on Google Colab. To reproduce locally:

```bash
pip install transformers datasets peft accelerate bitsandbytes torch tqdm matplotlib huggingface_hub
```

> Paths in the scripts (e.g. `/content/drive/MyDrive/...`) are Colab/Google Drive specific — adjust them to your own environment before running locally.

## Usage

### 1. Run the pipeline
Run the scripts/notebooks in order: `data-curation` → `fine-tune` → `convert-gguf`. Each stage writes its output to disk for the next stage to consume.

### 2. Inference with Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "<path-or-hf-repo-of-merged-model>"  # TODO: fill in your model path / HF Hub repo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto").eval()

messages = [{"role": "user", "content": "Perkenalkan dirimu"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

output = model.generate(input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.6)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

### 3. Inference with GGUF (llama.cpp)
```bash
./llama-cli -m sorachio-1b-q8_0.gguf -p "User: Perkenalkan dirimu\nSorachio:"
```

## Dataset & Weights

This repository only contains pipeline **code**. The curated dataset and trained model weights are not included here.

- Dataset: ~500k tokens of curated multi-turn Indonesian/multilingual dialogue (private).
- Model weights: (private).

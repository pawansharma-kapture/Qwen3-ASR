## Fine-tuning Qwen3-ASR

This script fine-tunes **Qwen3-ASR** using local JSONL files or a dataset from the Hugging Face Hub. It supports multi-GPU training via `torchrun` and optional dataset streaming for large datasets.

### 1) Setup

First, install `qwen-asr`, `datasets`, `pyyaml`, and `wandb`:

```bash
pip install -U qwen-asr datasets pyyaml wandb
```

To reduce GPU memory usage and speed up training, install FlashAttention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

FlashAttention 2 requires `torch.float16` or `torch.bfloat16`. See the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) for hardware requirements.

---

### 2) YAML config

All arguments can be set in a YAML file. CLI flags always override YAML values, so you can use the file as a base and tweak individual settings at launch.

A fully annotated template is provided at [`config.yaml`](./config.yaml). Copy and edit it:

```bash
cp config.yaml my_run.yaml
# edit my_run.yaml, then:
python qwen3_asr_sft.py --config my_run.yaml
# override a single value from CLI:
python qwen3_asr_sft.py --config my_run.yaml --lr 1e-5
```

---

### 3) Data sources

#### Option A — Local JSONL file

Prepare a JSONL file (one JSON object per line). Each line must have:

- `audio`: path to a WAV file
- `text`: transcript, with an optional language prefix

```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language English<asr_text>Another example."}
{"audio":"/data/wavs/utt0003.wav","text":"language None<asr_text>Unknown language sample."}
```

Language prefix guide:

| Situation | Prefix to use |
|---|---|
| Language known | `language English<asr_text>...` |
| Language known (Chinese) | `language Chinese<asr_text>...` |
| Language unknown | `language None<asr_text>...` |

> When `language None` is used, the model does not learn language detection from that sample.

#### Option B — Hugging Face Hub dataset

Your dataset must have an `audio` column (HF `Audio` feature) and a `text` column.

If your `text` column contains **plain transcriptions** (no language prefix), use `--language` to prepend it automatically:

```bash
python qwen3_asr_sft.py \
  --dataset_name "username/my-asr-dataset" \
  --language English \
  --dataset_train_split train \
  --dataset_eval_split test \
  --epochs 3
```

This prepends `language English<asr_text>` to every transcript at preprocessing time. If your `text` column already contains the prefix (e.g. `language English<asr_text>...`), omit `--language`.

Valid values follow the same convention as JSONL:

| `--language` value | Effect |
|---|---|
| `English` | Prepends `language English<asr_text>` |
| `Chinese` | Prepends `language Chinese<asr_text>` |
| `None` | Prepends `language None<asr_text>` (no language learning) |
| *(omitted)* | No prefix added — text is used as-is |

To stream the dataset (no full download — useful for very large datasets), add `--streaming` and set `--max_steps`:

```bash
python qwen3_asr_sft.py \
  --dataset_name "username/my-asr-dataset" \
  --language English \
  --streaming \
  --max_steps 10000 \
  --batch_size 8
```

> `--max_steps` is **required** in streaming mode because the dataset length is unknown.

---

### 4) Fine-tune (single GPU)

**From a local JSONL file:**

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

**From a HF Hub dataset:**

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --dataset_name "username/my-asr-dataset" \
  --dataset_eval_split test \
  --language English \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1
```

Checkpoints are saved to `./qwen3-asr-finetuning-out/checkpoint-<global_step>`.

---

### 5) Fine-tune (multi-GPU with torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

---

### 6) Auto eval split

If you have no separate eval set, use `--eval_split_ratio` to carve one out of the training data automatically. This uses `train_test_split()` after preprocessing, so the split is deterministic given the same seed.

```bash
python qwen3_asr_sft.py --config my_run.yaml --eval_split_ratio 0.05
```

This holds out 5% of the training data for evaluation and uses the remaining 95% for training. The split seed defaults to `42` and can be changed with `--eval_split_seed`.

Priority of eval sources (first match wins):
1. `--eval_file` / `--dataset_eval_split` — explicit eval set
2. `--eval_split_ratio` — auto-split from training data
3. *(none)* — no evaluation during training

> `--eval_split_ratio` is not compatible with `--streaming` since `IterableDataset` has no length.

### 8) Weights & Biases logging

Set `wandb_project` (in YAML or via CLI) to enable W&B run tracking:

```yaml
# in your config.yaml
wandb_project: my-asr-project
wandb_run_name: qwen3-1.7b-english-v1
```

Or pass directly:

```bash
python qwen3_asr_sft.py --config my_run.yaml \
  --wandb_project my-asr-project \
  --wandb_run_name qwen3-1.7b-english-v1
```

Make sure you are logged in first:

```bash
wandb login
```

### 7) Resume training

**Option A** — explicit checkpoint path:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

**Option B** — auto-resume from the latest checkpoint under `output_dir`:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

---

### 9) Quick inference test

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

---

### All arguments

> All of these can also be set in `config.yaml` using the same key names.

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-ASR-1.7B` | HF model ID or local path |
| `--train_file` | — | Path to JSONL training file |
| `--eval_file` | — | Path to JSONL eval file |
| `--output_dir` | `./qwen3-asr-finetuning-out` | Directory for checkpoints |
| `--dataset_name` | — | HF Hub dataset ID (e.g. `username/dataset`) |
| `--dataset_config` | — | HF dataset config/subset name |
| `--dataset_train_split` | `train` | Name of the training split |
| `--dataset_eval_split` | — | Name of the eval split |
| `--language` | — | Prepend `language LANG<asr_text>` to every transcript (e.g. `English`, `Chinese`, `None`) |
| `--streaming` | `False` | Stream the HF dataset (no full download) |
| `--sr` | `16000` | Audio sampling rate in Hz |
| `--batch_size` | `32` | Per-device training batch size |
| `--grad_acc` | `4` | Gradient accumulation steps |
| `--lr` | `2e-5` | Learning rate |
| `--epochs` | `1` | Number of training epochs |
| `--max_steps` | `-1` | Max training steps (required for `--streaming`) |
| `--log_steps` | `10` | Logging frequency in steps |
| `--lr_scheduler_type` | `linear` | LR scheduler type |
| `--warmup_ratio` | `0.02` | Warmup ratio |
| `--num_workers` | `4` | DataLoader worker processes |
| `--pin_memory` | `1` | Pin memory in DataLoader (`0` to disable) |
| `--persistent_workers` | `1` | Keep DataLoader workers alive between batches |
| `--prefetch_factor` | `2` | Batches prefetched per worker |
| `--save_strategy` | `steps` | Checkpoint save strategy |
| `--save_steps` | `200` | Save and eval every N steps |
| `--save_total_limit` | `5` | Max checkpoints to keep |
| `--eval_split_ratio` | `0.0` | Fraction of training data to use as eval (e.g. `0.05`). Ignored if eval source already provided. Not compatible with `--streaming`. |
| `--eval_split_seed` | `42` | Random seed for the auto eval split |
| `--resume_from` | — | Explicit checkpoint path to resume from |
| `--resume` | `0` | Set to `1` to auto-resume from latest checkpoint |
| `--wandb_project` | — | W&B project name (enables wandb logging when set) |
| `--wandb_run_name` | — | W&B run name |
| `--config` | — | Path to YAML config file |

---

### One-click shell script

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```

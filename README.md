# ScreenSpot GUI Grounding Evaluation

Evaluate vision-language models on the [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) and [ScreenSpot-Pro](https://huggingface.co/datasets/lscpku/ScreenSpot-Pro) GUI-grounding benchmarks.

Supports multiple models (Claude, Qwen 3 VL, Qwen 3.5 VL), multi-rollout voting, a two-pass zoom-in strategy, and train/validation holdout splitting for downstream RFT data generation.

## Setup

```bash
# Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create venv and install dependencies (eval only)
uv sync

# Or install with training dependencies (torch, accelerate, trl, etc.)
uv sync --extra train

# Configure API keys (copy .env.example or set directly)
cp .env.example .env
# Then fill in your keys in .env
```

### System dependencies for training

The training extras include CUDA extensions (`causal-conv1d`, `flash-linear-attention`) that compile from source. On a fresh GPU server you may need:

```bash
apt-get update && apt-get install -y python3-dev ninja-build
```

### Required environment variables

| Variable | When needed |
|---|---|
| `ANTHROPIC_API_KEY` | `--model claude` |
| `OPENROUTER_API_KEY` | `--model qwen3vl` or `--model qwen3.5vl` |
| `HF_TOKEN` | If gated datasets require authentication |
| `WANDB_API_KEY` | W&B logging during training (free at [wandb.ai](https://wandb.ai)) |

## Usage

```bash
# Run with uv
uv run python eval_screenspot.py [OPTIONS]
```

### Quick examples

```bash
# Quick test — 10 samples, default model (Qwen 3.5 VL)
uv run python eval_screenspot.py -n 10

# Full ScreenSpot benchmark (1272 samples)
uv run python eval_screenspot.py

# ScreenSpot-Pro dataset
uv run python eval_screenspot.py -m pro -n 50

# Use a specific model
uv run python eval_screenspot.py --model claude -n 20
uv run python eval_screenspot.py --model qwen3vl -n 20

# Generate train/validation split
uv run python eval_screenspot.py -n 200 --random-sample --sample-seed 42 \
    -o train.jsonl --holdout-output val.jsonl

# Two-pass zoom-in strategy
uv run python eval_screenspot.py -n 50 --zoom-in --crop-ratio 0.4
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model {claude,qwen3vl,qwen3.5vl}` | `qwen3.5vl` | Model backend to use |
| `-m, --mode {normal,pro}` | `normal` | Dataset variant |
| `-n, --num-samples N` | all | Limit number of samples |
| `--random-sample` | off | Sample uniformly at random |
| `--sample-seed SEED` | none | RNG seed for reproducible sampling |
| `-o, --output PATH` | `results.jsonl` | Output JSONL path |
| `--holdout-output PATH` | none | Write non-evaluated samples for validation |
| `-c, --concurrency N` | `10` | Max parallel API requests |
| `-r, --rollouts N` | `5` | Rollouts per sample |
| `-t, --temperature T` | `1.0` | Sampling temperature |
| `--zoom-in` | off | Enable two-pass zoom-in |
| `--crop-ratio R` | `0.5` | Zoom-in crop size (fraction of image) |

## Models

| Key | Model | Backend | Parameters |
|---|---|---|---|
| `claude` | Claude Sonnet 4.6 | Anthropic | — |
| `qwen3vl` | Qwen3-VL-235B-A22B | OpenRouter | 235B (22B active) |
| `qwen3.5vl` | Qwen3.5-122B-A10B | OpenRouter | 122B (10B active), native VL |

## SFT Training

Fine-tune Qwen3.5-4B on correct predictions from the 122B teacher model using LoRA + FSDP.

```bash
# 2-GPU FSDP training (recommended)
accelerate launch --config_file accelerate_config.yaml train_sft_qwen.py

# Single-GPU
python train_sft_qwen.py --batch-size 1 --grad-accum 16

# Custom W&B run name
accelerate launch --config_file accelerate_config.yaml train_sft_qwen.py \
    --wandb-project my-project --wandb-run-name "lora-r16-lr2e5"

# Disable W&B (TensorBoard only)
accelerate launch --config_file accelerate_config.yaml train_sft_qwen.py --no-wandb
```

### Training CLI reference

| Flag | Default | Description |
|---|---|---|
| `--results PATH` | `train_results.jsonl` | Results JSONL from 122B model evaluation |
| `-m, --mode {normal,pro}` | `pro` | ScreenSpot variant |
| `--deduplicate` | off | Keep one sample per unique image |
| `--model-id ID` | `Qwen/Qwen3.5-4B` | HuggingFace model ID |
| `--output-dir DIR` | `qwen35_4b_screenspot_lora` | Output directory for adapter |
| `--lora-r N` | `16` | LoRA rank |
| `--lora-alpha N` | `32` | LoRA alpha |
| `--lora-dropout F` | `0.05` | LoRA dropout |
| `--epochs N` | `3` | Training epochs |
| `--batch-size N` | `1` | Per-device batch size |
| `--grad-accum N` | `16` | Gradient accumulation steps |
| `--lr F` | `2e-5` | Learning rate |
| `--wandb-project NAME` | `screenspot-sft` | W&B project name |
| `--wandb-run-name NAME` | auto | W&B run name |
| `--wandb-entity NAME` | none | W&B team/user entity |
| `--no-wandb` | off | Disable W&B logging |

## Project structure

```
config/             Configuration split by concern
  models.py           Model definitions & registry
  datasets.py         HuggingFace dataset configs
  prompts.py          System prompt & CLICK regex
  defaults.py         Numeric defaults (concurrency, rollouts, etc.)

data/               Dataset loading
  sample.py           ScreenSpotSample dataclass & row parsers
  loader.py           load_screenspot() & load_screenspot_holdout()

inference/          Model inference
  image.py            Base64 encoding, crop, zoom-in mapping
  predict.py          predict_click() with Anthropic/OpenRouter dispatch

eval/               Evaluation orchestration
  client.py           API client factory & retry logic
  passes.py           Full-image pass & zoom-in pass runners
  runner.py           Top-level run_evaluation()

metrics.py          Accuracy, pass@k, majority vote, breakdown reporting
eval_screenspot.py  CLI entrypoint (evaluation)
train_sft_qwen.py  CLI entrypoint (SFT + LoRA training)
accelerate_config.yaml  FSDP multi-GPU config for accelerate
```

## Output format

Results are saved as JSONL, one record per sample-rollout:

```json
{
  "idx": 42,
  "rollout_id": 0,
  "file_name": "screen_042.png",
  "instruction": "Click the search bar",
  "bbox": [120, 45, 380, 75],
  "img_size": [1920, 1080],
  "data_type": "text",
  "data_source": "chrome",
  "predicted_coords_norm": [0.1302, 0.0556],
  "predicted_coords_px": [249.98, 60.05],
  "hit": true,
  "raw_response": "..."
}
```

The holdout file (`--holdout-output`) contains the same fields minus model outputs — just the metadata needed to reload samples for validation.

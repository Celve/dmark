# DMark - Diffusion LLM Watermarking System

A watermarking system for diffusion-based large language models supporting LLaDA and DREAM models.

## Installation

```bash
conda env create -f env.yml
conda activate dmark
```

## Quick Start

### 1. Generate Bitmap

```bash
python -m dmark.watermark.preprocess \
    --output_dir bitmaps/ \
    --vocab_size 126464 \
    --ratio 0.5 \
    --key 42
# Creates: bitmaps/bitmap_v126464_r50_k42.bin
```

### 2. Run Generation

**LLaDA Model:**
```bash
python -m dmark.llada.only_gen \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --strategy normal \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0 \
    --minimum_output_token 200
```

**DREAM Model:**
```bash
python -m dmark.llada.only_gen_dream \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --steps 512 \
    --alg entropy \
    --strategy normal \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0
```

## Batch Experiments

Use pre-configured experiment templates:

```bash
# Generate experiment script
python utils/generate_experiments.py utils/experiments/llada/baseline_no_watermark.json

# Run experiments
./run_baseline_no_watermark.sh

# Split into multiple scripts for parallel execution
python utils/generate_experiments.py \
    utils/experiments/llada/qa_watermark_strategies.json \
    --split 4
```

Available templates in `utils/experiments/llada/`:
- `baseline_no_watermark.json` - Baseline without watermarking
- `qa_watermark_strategies.json` - QA tasks with watermarking
- `text_watermark_strategies.json` - Text generation (C4)
- `code_watermark_strategies.json` - Code generation (HumanEval)
- `translation_watermark_strategies.json` - Translation (WMT16)
- `watermark_ratio_study.json` - Test different ratios
- `watermark_key_study.json` - Test different keys

## Evaluation

### Watermark Detection (Z-Score)
```bash
python -m dmark.eval.zscore \
    --input results.json \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --model GSAI-ML/LLaDA-8B-Instruct
```

### Text Quality (Perplexity)
```bash
python -m dmark.eval.ppl \
    --input results.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Robustness Testing
```bash
# Apply attack
python -m dmark.attack.attacks \
    --input results.json \
    --output attacked.json \
    --attack swap \
    --ratio 0.2

# Re-evaluate
python -m dmark.eval.zscore \
    --input attacked.json \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin
```

## Key Parameters

### Watermarking
- `--strategy`: `normal`, `predict`, `reverse`
- `--delta`: Watermark strength (1.0-10.0)
- `--ratio`: Green list ratio (0.25, 0.5, 0.75)
- `--key`: Hash key for randomization

### Generation (LLaDA)
- `--steps`: Denoising steps (default: 256)
- `--block_length`: Block size (default: 32)
- `--temperature`: Sampling temperature
- `--remasking`: `low_confidence`, `random`, `right_to_left`, `left_to_right`

### Generation (DREAM)
- `--steps`: Diffusion steps (default: 512)
- `--alg`: `origin`, `maskgit_plus`, `topk_margin`, `entropy`
- `--top_p`: Nucleus sampling (default: 0.95)
- `--temperature`: Sampling temperature (default: 0.2)

## Supported Datasets

- **QA**: `sentence-transformers/eli5` (default)
- **Text**: `allenai/c4`
- **Code**: `openai/openai_humaneval`
- **Translation**: `wmt16:de-en`, `wmt16:cs-en`, etc.

## Output Format

Results are saved as JSON with watermark detection scores and text quality metrics:

```json
{
    "data": {
        "prompt": "...",
        "output": "generated text",
        "output_ids": [...]
    },
    "watermark": {
        "z_score": 12.34,
        "detection_rate": 0.75
    },
    "text_quality": {
        "perplexity": 15.67
    }
}
```

## Tips

1. **Filter short outputs**: Use `--minimum_output_token 200` to exclude incomplete generations

2. **Batch processing**: Create custom experiment JSON files for parameter sweeps

3. **Parallel execution**: Use `--split N` to distribute experiments across multiple machines

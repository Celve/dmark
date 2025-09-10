# DMark - Diffusion LLM Watermarking System

A watermarking system for diffusion-based large language models supporting LLaDA and DREAM models.

## Installation

```bash
conda env create -f env.yml
conda activate dmark
```

## Quick Start

### 1. Generate Bitmap

For Llada: 

```bash
python -m dmark.watermark.preprocess \
    --output_dir bitmaps/ \
    --vocab_size 126464 \
    --ratio 0.5 \
    --key 42
# Creates: bitmaps/bitmap_v126464_r50_k42.bin
```

For Dream 7B: 

```bash
python -m dmark.watermark.preprocess \
    --output_dir bitmaps/ \
    --vocab_size 152064 \
    --ratio 0.5 \
    --key 42
# Creates: bitmaps/bitmap_v152064_r50_k42.bin
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

### Generate and Run Experiments

```bash
# Generate experiment script (saved to scripts/ directory)
python utils/generate_experiments.py utils/experiments/llada/baseline_no_watermark.json

# Run experiments
./scripts/run_baseline_no_watermark.sh

# Split into multiple scripts for parallel execution
python utils/generate_experiments.py \
    utils/experiments/llada/qa_watermark_strategies.json \
    --split 4

# Run split scripts in parallel
./scripts/run_qa_watermark_strategies_part1_of_4.sh  # Terminal 1
./scripts/run_qa_watermark_strategies_part2_of_4.sh  # Terminal 2
# etc.
```

### Available Experiment Templates

Located in `utils/experiments/llada/`:

- `baseline_no_watermark.json` - Baseline without watermarking
- `qa_watermark_strategies.json` - QA tasks with watermarking
- `text_watermark_strategies.json` - Text generation (C4)
- `code_watermark_strategies.json` - Code generation (HumanEval)
- `math_watermark_strategies.json` - Math problem solving (GSM8K)
- `translation_watermark_strategies.json` - Translation (WMT16)
- `watermark_ratio_study.json` - Test different ratios
- `watermark_key_study.json` - Test different keys
- `generation_order_effects.json` - Generation order vs watermark strategies

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

### Threshold Analysis
```bash
python -m dmark.analysis.threshold_calculator \
    --input_dir results/ \
    --pattern "*_zscore.json" \
    --output threshold_analysis.json
```

## Utility Tools

### Convert Threshold Results to CSV
```bash
# Simple conversion (filename and thresholds only)
python utils/json_to_csv_simple.py threshold_analysis.json

# With metadata extraction
python utils/json_to_csv_thresholds.py threshold_analysis.json --metadata
```

### Experiment Generator Options
```bash
# Custom output file
python utils/generate_experiments.py config.json --output my_script.sh

# Dry run to preview commands
python utils/generate_experiments.py config.json --dry-run

# Limit number of commands
python utils/generate_experiments.py config.json --max-commands 10
```

## Key Parameters

### Watermarking
- `--strategy`: Watermark strategy
  - `normal`: Uses previous token (i-1)
  - `predict`: Uses predicted tokens
  - `reverse`: Bidirectional (both prev and next)
- `--delta`: Watermark strength (1.0-10.0)
- `--ratio`: Green list ratio (0.25, 0.5, 0.75)
- `--key`: Hash key for randomization
- `--prebias`: Apply watermark before token selection

### Generation (LLaDA)
- `--steps`: Denoising steps (default: 256)
- `--block_length`: Block size for semi-autoregressive generation (default: 32)
- `--temperature`: Sampling temperature (0.0 for greedy)
- `--cfg_scale`: Classifier-free guidance scale
- `--remasking`: Token revelation order
  - `low_confidence`: Unmask least confident first
  - `random`: Random order
  - `right_to_left`: Reverse sequential
  - `left_to_right`: Traditional sequential

### Generation (DREAM)
- `--steps`: Diffusion steps (default: 512)
- `--alg`: Sampling algorithm
  - `origin`: Original DREAM sampling
  - `maskgit_plus`: MaskGIT-inspired
  - `topk_margin`: Top-k with margin
  - `entropy`: Entropy-based
- `--top_p`: Nucleus sampling (default: 0.95)
- `--temperature`: Sampling temperature (default: 0.2, **Note: Must be > 0 for DREAM models**)
- `--alg_temp`: Algorithm-specific temperature
- `--eps`: Epsilon for numerical stability
- `--top_k`: Top-k sampling parameter

## Supported Datasets

- **QA**: `sentence-transformers/eli5` (default)
- **Text**: `allenai/c4`
- **Code**: `openai/openai_humaneval`
- **Math**: `gsm8k` or `openai/gsm8k`
- **Translation**: `wmt16:de-en`, `wmt16:cs-en`, `wmt16:fi-en`, `wmt16:ro-en`, `wmt16:ru-en`, `wmt16:tr-en`

## Output Format

### Generation Results
```json
{
    "data": {
        "prompt": "...",
        "ground_truth": "...",
        "output": "generated text",
        "output_ids": [...],
        "num_output_tokens": 256
    },
    "generation_metadata": {
        "model": "GSAI-ML/LLaDA-8B-Instruct",
        "dataset": "sentence-transformers/eli5",
        "steps": 256,
        "gen_length": 256,
        "block_length": 32,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence"
    },
    "watermark_metadata": {
        "vocab_size": 126464,
        "ratio": 0.5,
        "delta": 2.0,
        "key": 42,
        "prebias": false,
        "strategy": "normal",
        "bitmap_path": "bitmaps/bitmap_v126464_r50_k42.bin"
    }
}
```

### Z-Score Results
```json
{
    "z_score": 12.34,
    "detection_rate": 0.75,
    "num_green_tokens": 192,
    "total_tokens": 256
}
```

## Directory Structure

```
dmark/
├── dmark/                    # Main package
│   ├── llada/               # Generation modules
│   │   ├── only_gen.py      # LLaDA generation
│   │   └── only_gen_dream.py # DREAM generation
│   ├── watermark/           # Watermarking modules
│   │   ├── watermark.py     # Core watermark class
│   │   ├── config.py        # Configuration
│   │   └── preprocess.py    # Bitmap generation
│   ├── eval/                # Evaluation tools
│   │   ├── zscore.py        # Z-score detection
│   │   └── ppl.py           # Perplexity evaluation
│   ├── attack/              # Robustness testing
│   └── analysis/            # Analysis tools
├── utils/                   # Utility scripts
│   ├── generate_experiments.py
│   ├── json_to_csv_simple.py
│   ├── json_to_csv_thresholds.py
│   └── experiments/         # Experiment configs
├── bitmaps/                 # Generated bitmaps (gitignored)
├── results/                 # Experiment results (gitignored)
└── scripts/                 # Generated scripts (gitignored)
```

## Tips

1. **Bitmap Naming**: Bitmaps are automatically named as `bitmap_v{vocab_size}_r{ratio*100}_k{key}.bin`

2. **Filter Short Outputs**: Use `--minimum_output_token 200` to exclude incomplete generations

3. **Batch Processing**: Create custom experiment JSON files for parameter sweeps

4. **Parallel Execution**: Use `--split N` to distribute experiments across multiple GPUs/machines

5. **Result Organization**: Output filenames include all parameters for easy identification

6. **Memory Management**: Use streaming for large datasets like C4

## License

[Your License Here]
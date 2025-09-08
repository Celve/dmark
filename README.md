# DMark - Diffusion LLM Watermarking System

DMark is a watermarking system for diffusion-based large language models (DLLMs) that implements LLaDA-based generation with watermark injection and detection capabilities. The system supports both LLaDA and DREAM models for watermarked text generation.

## Installation

```bash
# Create conda environment
conda env create -f env.yml
conda activate dmark
```

## Quick Start

### 1. Generate Watermark Bitmap

Before using watermarking, you must first generate a bitmap file:

```bash
# Generate bitmap with default parameters
python -m dmark.watermark.preprocess --output_dir bitmaps/

# Custom parameters
python -m dmark.watermark.preprocess \
    --output_dir bitmaps/ \
    --vocab_size 126464 \
    --ratio 0.5 \
    --key 42
```

This creates a bitmap file named `bitmap_v{vocab_size}_r{ratio*100}_k{key}.bin` (e.g., `bitmap_v126464_r50_k42.bin`).

### 2. Generate Watermarked Text

#### Using LLaDA Model

```bash
# Basic generation with watermark
python -m dmark.llada.only_gen \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --strategy reverse \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0 \
    --minimum_output_token 200

# Generation without watermark (baseline)
python -m dmark.llada.only_gen \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256
```

#### Using DREAM Model

```bash
# DREAM model with watermark
python -m dmark.llada.only_gen_dream \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --steps 512 \
    --temperature 0.2 \
    --alg entropy \
    --top_p 0.95 \
    --strategy normal \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0

# DREAM model without watermark
python -m dmark.llada.only_gen_dream \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --steps 512 \
    --alg entropy
```

## Supported Datasets

DMark supports multiple dataset formats for watermark generation experiments:

### 1. Question-Answering Datasets (Default)

**Example**: `sentence-transformers/eli5`

**Usage**: Questions are used as prompts with chat template formatting, answers serve as ground truth reference.

```bash
python -m dmark.llada.only_gen \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256
```

### 2. C4 Dataset (Text Completion)

**Dataset**: `allenai/c4`

**Usage**: First 30 tokens from C4 text serve as prompts, next `gen_length` tokens (e.g., 200) serve as non-watermarked ground truth. The model generates `gen_length` tokens to complete the text.

```bash
python -m dmark.llada.only_gen \
    --dataset allenai/c4 \
    --num_samples 100 \
    --gen_length 200  # Will use tokens 30-230 as ground truth
```

**Note**: Uses streaming mode for memory efficiency. Texts shorter than 30 + `gen_length` tokens are automatically skipped.

### 3. HumanEval (Code Generation)

**Dataset**: `openai/openai_humaneval`

**Usage**: Function signatures with docstrings serve as prompts. The model generates code completions. Canonical solutions are stored as reference but not used for constraining generation.

```bash
python -m dmark.llada.only_gen \
    --dataset openai/openai_humaneval \
    --num_samples 100 \
    --gen_length 256
```

**Note**: Contains 164 programming problems. Uses test split only.

### 4. WMT16 (Machine Translation)

**Dataset**: `wmt16:lang_pair` where lang_pair is one of: `cs-en`, `de-en`, `fi-en`, `ro-en`, `ru-en`, `tr-en`

**Usage**: Source language text serves as prompt, target language translation as ground truth reference. The model generates translations.

```bash
# German to English translation
python -m dmark.llada.only_gen \
    --dataset wmt16:de-en \
    --num_samples 100 \
    --gen_length 256

# Russian to English translation
python -m dmark.llada.only_gen \
    --dataset wmt16:ru-en \
    --num_samples 100 \
    --gen_length 256
```

### Dataset Format Detection

The system automatically detects dataset format based on the dataset name:
- `allenai/c4` → Text completion mode
- `openai/openai_humaneval` → Code generation mode
- `wmt16:*` → Translation mode
- Others → Question-answering mode (default)

## Evaluation Scripts

The `dmark/eval` directory contains scripts for evaluating watermarked text quality and detection performance.

### Z-Score Calculation (Watermark Detection)

Calculate z-scores to detect watermarked content:

```bash
# Single file
python -m dmark.eval.zscore \
    --input results.json \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --model GSAI-ML/LLaDA-8B-Instruct

# Process entire directory
python -m dmark.eval.zscore \
    --input results/ \
    --output zscore_results/ \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin
```

**Output**: Adds a `watermark` field with z-score and detection rate to each entry.

### Perplexity Evaluation

Measure text quality using perplexity:

```bash
# Single file
python -m dmark.eval.ppl \
    --input results.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Process directory with custom output location
python -m dmark.eval.ppl \
    --input results/ \
    --output ppl_results/ \
    --device cuda
```

**Output**: Adds perplexity score under `text_quality.perplexity`.

### Log Diversity Analysis

Evaluate text diversity based on n-gram uniqueness:

```bash
# Single file
python -m dmark.eval.log_diversity \
    --input results.json

# Process directory
python -m dmark.eval.log_diversity \
    --input results/ \
    --output diversity_results/
```

**Output**: Adds log diversity score under `text_quality.log_diversity`.

### BLEU Score Calculation

Compare attacked text with original (useful after attacks):

```bash
# For attacked files (compares with original_output)
python -m dmark.eval.bleu \
    --input attacked_results.json

# Custom reference field
python -m dmark.eval.bleu \
    --input results/ \
    --output bleu_results/ \
    --reference_field reference
```

**Output**: Adds BLEU score under `text_quality.bleu`.

### Combined Evaluation Pipeline

You can run multiple evaluations in sequence:

```bash
# 1. Generate watermarked text
python -m dmark.llada.only_gen \
    --dataset sentence-transformers/eli5 \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --num_samples 1000 \
    --output_dir results/ \
    --strategy reverse \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0

# 2. Calculate z-scores
python -m dmark.eval.zscore \
    --input results/ \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin

# 3. Evaluate text quality
python -m dmark.eval.ppl --input results/
python -m dmark.eval.log_diversity --input results/

# 4. Results will have all metrics in the JSON files
```

## Experiment Generation System

The `utils/generate_experiments.py` script automates the creation of experiment scripts from JSON configuration files.

### Configuration Files

Pre-configured experiment templates are available in `utils/experiments/llada/`:

- `baseline_no_watermark.json` - Baseline experiments without watermarking
- `qa_watermark_strategies.json` - QA tasks with different watermarking strategies
- `text_watermark_strategies.json` - Text generation (C4) experiments
- `code_watermark_strategies.json` - Code generation (HumanEval) experiments
- `translation_watermark_strategies.json` - Translation (WMT16) experiments
- `watermark_ratio_study.json` - Study different green list ratios
- `watermark_key_study.json` - Study different hash keys

### Basic Usage

```bash
# Generate experiment script from configuration
python utils/generate_experiments.py utils/experiments/llada/baseline_no_watermark.json

# This creates: run_baseline_no_watermark.sh
./run_baseline_no_watermark.sh
```

### Advanced Features

#### 1. Split Experiments Across Multiple Scripts

Split large experiments into N independent scripts for parallel execution:

```bash
# Split into 4 separate scripts
python utils/generate_experiments.py \
    utils/experiments/llada/qa_watermark_strategies.json \
    --split 4

# Creates:
# - run_qa_watermark_strategies_part1_of_4.sh
# - run_qa_watermark_strategies_part2_of_4.sh
# - run_qa_watermark_strategies_part3_of_4.sh
# - run_qa_watermark_strategies_part4_of_4.sh

# Run in parallel (different terminals/machines)
./run_qa_watermark_strategies_part1_of_4.sh  # Terminal 1
./run_qa_watermark_strategies_part2_of_4.sh  # Terminal 2
./run_qa_watermark_strategies_part3_of_4.sh  # Terminal 3
./run_qa_watermark_strategies_part4_of_4.sh  # Terminal 4
```

#### 2. Preview Commands (Dry Run)

```bash
# Preview generated commands without creating script
python utils/generate_experiments.py \
    utils/experiments/llada/watermark_ratio_study.json \
    --dry-run

# Preview with splitting
python utils/generate_experiments.py \
    utils/experiments/llada/watermark_ratio_study.json \
    --split 3 \
    --dry-run
```

#### 3. Custom Output Files

```bash
# Specify custom output filename
python utils/generate_experiments.py \
    utils/experiments/llada/code_watermark_strategies.json \
    --output my_code_experiments.sh
```

#### 4. Limit Number of Experiments

```bash
# Generate only first 10 experiments (useful for testing)
python utils/generate_experiments.py \
    utils/experiments/llada/text_watermark_strategies.json \
    --max-commands 10
```

### Creating Custom Experiment Configurations

Create a JSON file with the following structure:

```json
{
    "name": "my_experiment",
    "description": "Description of the experiment",
    "base_command": "python -m dmark.llada.only_gen",
    "fixed_params": {
        "model": "GSAI-ML/LLaDA-8B-Instruct",
        "output_dir": "results/my_experiment",
        "vocab_size": 126464,
        "num_samples": 500,
        "steps": 256,
        "gen_length": 256,
        "bitmap_dir": "."
    },
    "variable_params": {
        "dataset": ["sentence-transformers/eli5", "allenai/c4"],
        "strategy": ["normal", "predict", "reverse"],
        "delta": [1.0, 2.0, 5.0]
    }
}
```

The script will generate all combinations of variable parameters (2 × 3 × 3 = 18 experiments in this example).

### Automatic Bitmap Naming

The experiment generator automatically constructs bitmap filenames based on `vocab_size`, `ratio`, and `key`:

```json
{
    "fixed_params": {
        "vocab_size": 126464,
        "ratio": 0.5,
        "key": 42,
        "bitmap_dir": "bitmaps"
    }
}
```

This automatically generates: `--bitmap bitmaps/bitmap_v126464_r50_k42.bin`

## Analysis Scripts

The `dmark/analysis` directory contains scripts for analyzing watermark detection thresholds.

### Threshold Calculator

Determine z-score thresholds for target true positive rates:

```bash
# Analyze watermarked samples for threshold determination
python -m dmark.analysis.threshold_calculator \
    --input zscore_results/ \
    --tpr 0.90 0.95 0.99 0.999
```

**Features**:
- Processes each file separately to show consistency
- Calculates thresholds for 90%, 95%, 99%, and 99.9% detection rates
- Shows actual vs. target TPR
- Provides detailed statistics and percentiles

**Output**: 
- `threshold_analysis_per_file.json` with per-file results
- Console output showing thresholds and statistics

### Example Analysis Workflow

```bash
# 1. Generate watermarked samples
python -m dmark.llada.only_gen \
    --dataset sentence-transformers/eli5 \
    --num_samples 1000 \
    --output_dir results/ \
    --strategy normal \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 5.0

# 2. Calculate z-scores
python -m dmark.eval.zscore \
    --input results/ \
    --output zscore_results/ \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin

# 3. Determine detection thresholds
python -m dmark.analysis.threshold_calculator \
    --input zscore_results/ \
    --tpr 0.90 0.95 0.99 0.999
```

## Attack Scripts

The `dmark/attack` directory contains scripts for testing watermark robustness.

### Random Swap Attack

Randomly swap token pairs (default: 20%):

```bash
# Single file
python -m dmark.attack.attacks \
    --input results.json \
    --output attacked.json \
    --attack swap \
    --ratio 0.2

# Process directory
python -m dmark.attack.attacks \
    --input results/ \
    --output attacked_results/ \
    --attack swap \
    --ratio 0.3 \
    --seed 42
```

### Random Delete Attack

Randomly delete tokens (default: 20%):

```bash
# Apply deletion attack
python -m dmark.attack.attacks \
    --input results/ \
    --output deleted_results/ \
    --attack delete \
    --ratio 0.2
```

### Evaluate Attack Impact

After applying attacks, evaluate the impact:

```bash
# 1. Apply attack
python -m dmark.attack.attacks \
    --input results/ \
    --output attacked/ \
    --attack swap \
    --ratio 0.2

# 2. Calculate z-scores on attacked text
python -m dmark.eval.zscore \
    --input attacked/ \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin

# 3. Compare with original using BLEU
python -m dmark.eval.bleu --input attacked/

# 4. Analyze detection rates
python -m dmark.analysis.threshold_calculator --input attacked/
```

## Output Format

All scripts save results in JSON format with the following structure:

```json
{
    "data": {
        "prompt": "...",
        "output": "generated text",
        "output_ids": [token_ids],
        "original_output": "..."  // For attacked files
    },
    "watermark_metadata": {
        "vocab_size": 126464,
        "ratio": 0.5,
        "delta": 2.0,
        "key": 42,
        "strategy": "normal"
    },
    "watermark": {
        "z_score": 12.34,
        "detection_rate": 0.75
    },
    "text_quality": {
        "perplexity": 15.67,
        "log_diversity": 3.45,
        "bleu": 85.23
    },
    "attack_metadata": {
        "type": "swap",
        "ratio": 0.2,
        "original_length": 256,
        "attacked_length": 256
    }
}
```

## Generation Parameters

### Minimum Output Token

The `--minimum_output_token` parameter filters out samples with insufficient output length:

```bash
# Only keep samples with at least 200 output tokens
python -m dmark.llada.only_gen \
    --dataset sentence-transformers/eli5 \
    --num_samples 1000 \
    --gen_length 256 \
    --minimum_output_token 200 \
    --output_dir results/
```

This ensures high-quality samples for watermark detection analysis by excluding truncated or incomplete generations.

## Watermarking Strategies

- **normal**: Basic watermark based on previous token (i-1)
- **predict**: Uses predicted tokens for watermarking
- **reverse**: Bidirectional watermarking using both previous and next tokens
- **legacy-ahead**: Legacy compatibility mode (ahead prediction)
- **legacy-both**: Legacy compatibility mode (bidirectional)

## Generation Strategies

### LLaDA Model Parameters

- **steps**: Number of denoising steps (default: 256)
- **block_length**: Block size for semi-autoregressive generation (default: 32)
- **temperature**: Sampling temperature (0.0 for greedy)
- **cfg_scale**: Classifier-free guidance scale
- **remasking**: Strategy for remasking tokens
  - **low_confidence**: Remask tokens with lowest confidence scores
  - **random**: Randomly assign confidence scores
  - **right_to_left**: Remask from right to left (rightmost tokens first)
  - **left_to_right**: Remask from left to right (leftmost tokens first)

### DREAM Model Parameters

- **steps**: Number of diffusion steps (default: 512)
- **alg**: Sampling algorithm
  - **origin**: Original DREAM sampling
  - **maskgit_plus**: MaskGIT+ algorithm
  - **topk_margin**: Top-k margin confidence
  - **entropy**: Entropy-based sampling
- **alg_temp**: Algorithm temperature
- **top_p**: Top-p (nucleus) sampling
- **top_k**: Top-k sampling
- **eps**: Epsilon for numerical stability

## Tips

1. **Bitmap Generation**: Generate bitmaps with different parameters for experiments:
   ```bash
   for key in 42 123 456; do
       python -m dmark.watermark.preprocess --output_dir bitmaps/ --key $key
   done
   ```

2. **Batch Processing**: Process multiple configurations:
   ```bash
   for delta in 1.0 2.0 5.0 10.0; do
       for strategy in normal predict reverse; do
           python -m dmark.llada.only_gen \
               --output_dir results/delta_${delta}_${strategy}/ \
               --delta $delta \
               --strategy $strategy \
               --bitmap bitmaps/bitmap_v126464_r50_k42.bin
       done
   done
   ```

3. **Full Pipeline**: Complete evaluation pipeline:
   ```bash
   # Generate → Evaluate → Attack → Re-evaluate
   DIR=experiment_001
   mkdir -p $DIR
   
   # Generate
   python -m dmark.llada.only_gen --output_dir $DIR/original/ ...
   
   # Evaluate
   python -m dmark.eval.zscore --input $DIR/original/ --output $DIR/zscore/
   python -m dmark.eval.ppl --input $DIR/original/ --output $DIR/quality/
   
   # Attack
   python -m dmark.attack.attacks --input $DIR/original/ --output $DIR/attacked/
   
   # Re-evaluate
   python -m dmark.eval.zscore --input $DIR/attacked/ --output $DIR/attacked_zscore/
   python -m dmark.eval.bleu --input $DIR/attacked/
   
   # Analyze
   python -m dmark.analysis.threshold_calculator --input $DIR/zscore/
   python -m dmark.analysis.threshold_calculator --input $DIR/attacked_zscore/
   ```

## Citation

If you use DMark in your research, please cite:

```bibtex
@software{dmark2024,
  title = {DMark: Discrete LLM Watermarking System},
  year = {2024},
  url = {https://github.com/yourusername/dmark}
}
```

## License

[Your License Here]
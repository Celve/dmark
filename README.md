# DMark - Diffusion LLM Watermarking System

DMark is a watermarking system for diffusion-based large language models (DLLMs) that implements LLaDA-based generation with watermark injection and detection capabilities.

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

This creates a bitmap file named `bitmap_v{vocab_size}_r{ratio}_k{key}.bin`.

### 2. Generate Watermarked Text

```bash
# Basic generation with watermark
python -m dmark.llada.gen \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256 \
    --strategy reverse \
    --bitmap bitmaps/bitmap_v126464_r50_k42.bin \
    --delta 2.0

# Generation without watermark (baseline)
python -m dmark.llada.gen \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --dataset sentence-transformers/eli5 \
    --num_samples 100 \
    --gen_length 256
```

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

## Watermarking Strategies

- **normal**: Basic watermark based on previous token (i-1)
- **predict**: Uses predicted tokens for watermarking
- **reverse**: Bidirectional watermarking using both previous and next tokens

## Remasking Strategies

- **low_confidence**: Remask tokens with lowest confidence scores
- **random**: Randomly assign confidence scores
- **right_to_left**: Remask from right to left (rightmost tokens first)
- **left_to_right**: Remask from left to right (leftmost tokens first)

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
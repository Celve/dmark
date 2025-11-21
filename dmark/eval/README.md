# Eval Utilities

Command-line helpers for annotating generation results.

## Perplexity (`ppl.py`)
- Attach `{ "perplexity": value }` under a configurable key (default `text_quality`) for each sample.
- Works on a file or directory; supports `--increment` to skip already-tagged outputs.
- Common flags: `--input`, `--output` / `--tag`, `--model`, `--device auto|cpu|cuda`, `--insert-key`.

Examples
```bash
# Single file, default output beside input
python -m dmark.eval.ppl --input results.json

# Directory, skip existing tagged outputs, write to custom folder
python -m dmark.eval.ppl \
  --input results_dir \
  --output annotated_dir \
  --increment
```

## Watermark Detection (`watermark.py`)
- Runs `detect()` on each sample’s tokens and inserts the returned dict (default key `watermark_detection`).
- Automatically uses `watermark_metadata` inside each record when present; otherwise falls back to CLI args.
- For bitmap strategies, derives the bitmap path from `--bitmap-dir`, `--vocab_size`, `--ratio`, and `--key`.

Examples
```bash
# File-level
python -m dmark.eval.watermark \
  --input results.json \
  --bitmap-dir bitmaps \
  --strategy normal

# Directory-level
python -m dmark.eval.watermark \
  --input results_dir \
  --output-dir annotated_dir \
  --bitmap-dir bitmaps \
  --strategy pattern-mark
```

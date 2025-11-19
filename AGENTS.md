# Repository Guidelines

## Project Structure & Module Organization
`dmark/` hosts runtime code: `watermark/` (bitmap configs + persistence), `llada/` (LLaDA & DREAM entrypoints), `eval/` (z-score/perplexity), `attack/` (robustness sweeps), and `analysis/`. Shared helpers and experiment templates live in `utils/` (see `generate_experiments.py` and `experiments/{llada,dream,llada-1.5}/`). Versioned configs belong in `configs/`; generated artifacts (`bitmaps/`, `results/`, `scripts/`) stay gitignored.

## Build, Test, and Development Commands
1. `conda env create -f env.yml && conda activate dmark` – provision Python 3.10 + torch/transformers toolchain.
2. `python -m dmark.watermark.preprocess --output_dir bitmaps/ --vocab_size 126464 --ratio 0.5 --key 42` – rebuild canonical bitmap before regenerations.
3. `python -m dmark.llada.only_gen --model GSAI-ML/LLaDA-8B-Instruct --dataset sentence-transformers/eli5 --num_samples 32 --gen_length 256 --bitmap bitmaps/...bin` – smoke-test LLaDA; swap to `only_gen_dream` for DREAM runs.
4. `python utils/generate_experiments.py utils/experiments/llada/qa_watermark_strategies.json --split 4` then run emitted `scripts/run_*part*.sh` files to parallelize sweeps.
5. `python -m dmark.eval.zscore --input results/smoke.json --bitmap bitmaps/...bin` and `python -m dmark.eval.ppl --input results/smoke.json --model meta-llama/Meta-Llama-3-8B-Instruct` – regression checks required before opening a PR.
6. `bash tpr-fpr.sh` – optional exhaustive matrix for remasking/ratio tuning; edit N/min_token inline.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indents, 100-column soft limit, and snake_case modules/flags; classes stay CamelCase (`WatermarkConfig`). Type-hint public functions and keep argparse definitions near entrypoints (see `dmark/watermark/preprocess.py`). Name bitmaps `bitmap_v{vocab}_r{ratio*100}_k{key}.bin`, experiment configs `*_strategies.json`, and results `results/{model}_{dataset}_{timestamp}.json`.

## Testing Guidelines
There is no pytest suite; verification relies on CLI evaluations. Before pushing, run a short LLaDA or DREAM job plus the paired `eval.zscore`/`eval.ppl` commands and record detection/perplexity deltas inside your PR. For new attack routines, add a JSON template under `utils/experiments/<domain>/smoke_<feature>.json` and reference it from the PR so reviewers can rerun the same scenario. Keep raw outputs under `results/` (gitignored) and attach aggregated metrics or plots instead.

## Commit & Pull Request Guidelines
Use Conventional Commits (`feat:`, `fix:`, `docs:`) as seen in `feat: support ppl pipeline`. Squash unrelated changes and mention affected modules in the subject. Every PR should describe the dataset/model pair exercised, link any experiment scripts, and paste the latest z-score/perplexity snapshot in a markdown table. Request at least one reviewer familiar with the touched module and have CI artifacts or log snippets ready for follow-up.

## Security & Configuration Tips
Never commit model weights, Hugging Face tokens, or generated `bitmaps/`/`results/`. Store credentials in your shell profile, and prefer relative paths so shared scripts (e.g., `tpr-fpr.sh`) remain portable across clusters.

# View Helpers

Lightweight, read-only utilities for inspecting result JSON files without mutating them.

## Philosophy

- **No writes:** Functions operate in-memory and do not persist modified files.
- **Composable callbacks:** You supply an `aggregate` callable that receives each JSON record (`dict`) and can accumulate metrics, log, or stream elsewhere.
- **Minimal dependencies:** Standard library only.

## Modules

### process.py

Core traversal helpers:

- `process_file(input_path: Path, aggregate: Callable[[dict], Any]) -> list[Any]`
  - Reads a JSON file. Runs `aggregate` on each dict record and returns the list of aggregate returns.
  - Non-dict items are ignored.

- `process_dir(input_dir: Path, aggregate: Callable[[dict], Any]) -> dict[Path, list[Any]]`
  - Iterates `.json` files (non-recursive) and returns a mapping from file path to the list of aggregate returns for that file.

Usage sketch:

```python
from pathlib import Path
from dmark.view.process import process_dir

total = 0
count = 0

def aggregate(record: dict):
    global total, count
    scores = [
        record.get("watermark", {}).get(k)
        for k in ("z_score_original", "z_score_truncated", "z_score_attacked", "z_score")
    ]
    for z in scores:
        if isinstance(z, (int, float)):
            total += z
            count += 1

process_dir(Path("results_dir"), aggregate)
print("avg z:", total / count if count else "n/a")
```

## When to use

- Ad-hoc inspection (metrics, counts) without rewriting artifacts.
- Quick sanity checks in pipelines where outputs must stay immutable.
- Building small one-off scripts by composing `process_file` / `process_dir` with custom callbacks.

### zscore.py

Reference CLI built on `process.py`; prints average z-scores per file for a file or directory input.

### fpr.py

Computes z-score quantile thresholds (e.g., 0.999 / 0.99) per generation config present in a file or directory. Outputs to stdout or JSON via `--output`.

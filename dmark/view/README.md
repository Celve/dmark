# View Helpers

Lightweight, read-only utilities for inspecting result JSON files without mutating them.

## Philosophy

- **No writes:** Functions operate in-memory and do not persist modified files.
- **Composable callbacks:** You supply an `aggregate` callable that receives each JSON record (`dict`) and can accumulate metrics, log, or stream elsewhere.
- **Minimal dependencies:** Standard library only.

## Modules

### process.py

Core traversal helpers:

- `process_file(input_path: Path, aggregate: Callable[[dict], Any]) -> None`
  - Reads a JSON file. If it contains a list, the callback runs on every dict item; if it is a single dict, the callback runs once.
  - Non-dict items are ignored.
  - Returns nothing; side effects belong entirely to `aggregate`.

- `process_dir(input_dir: Path, aggregate: Callable[[dict], Any]) -> None`
  - Iterates `.json` files (non-recursive) in a directory and invokes `process_file` on each.

Usage sketch:

```python
from pathlib import Path
from dmark.view.process import process_dir

total = 0
count = 0

def aggregate(record: dict):
    global total, count
    z = record.get("watermark", {}).get("z_score")
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

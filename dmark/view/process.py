import json
from pathlib import Path
from typing import Callable, Any


def process_file(input_path: Path, aggregate: Callable[[dict], Any]) -> list[Any]:
    """Read a JSON file and apply ``aggregate`` to each dict item, returning all results."""
    input_path = Path(input_path)
    with input_path.open("r") as f:
        data = json.load(f)

    results: list[Any] = []
    if isinstance(data, list):
        for item in data:
            results.append(aggregate(item))
    else:
        results.append(aggregate(data))
    return results


def process_dir(input_dir: Path, aggregate: Callable[[dict], Any]) -> dict[Path, list[Any]]:
    """Apply ``aggregate`` to every record in each .json file in ``input_dir``."""
    input_dir = Path(input_dir)
    results: dict[Path, list[Any]] = {}
    for path in input_dir.iterdir():
        if path.is_dir() or path.suffix.lower() != ".json":
            continue
        results[path] = process_file(path, aggregate)
    return results

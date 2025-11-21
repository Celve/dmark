import json
from pathlib import Path
from typing import Callable, Any


def _apply_instance(instance: Any, aggregate: Callable[[dict], Any]) -> None:
    if isinstance(instance, dict):
        aggregate(instance)


def process_file(input_path: Path, aggregate: Callable[[dict], Any]) -> None:
    """Read a JSON file and apply ``aggregate`` to each dict item, discarding output."""
    input_path = Path(input_path)
    with input_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            _apply_instance(item, aggregate)
    else:
        _apply_instance(data, aggregate)


def process_dir(input_dir: Path, aggregate: Callable[[dict], Any]) -> None:
    """Apply ``aggregate`` to every record in each .json file in ``input_dir``."""
    input_dir = Path(input_dir)
    for path in input_dir.iterdir():
        if path.is_dir() or path.suffix.lower() != ".json":
            continue
        process_file(path, aggregate)

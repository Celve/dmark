import json
from pathlib import Path
from typing import Callable, Any, Iterable

from tqdm import tqdm


def _process_instance(instance: Any, transform: Callable[[dict], dict], insert_key: str) -> Any:
    if not isinstance(instance, dict):
        return instance
    derived = transform(instance)
    instance[insert_key] = derived
    return instance


def process_file(
    input_path: Path,
    transform: Callable[[dict], dict],
    *,
    insert_key: str = "processed",
    output_path: Path | None = None,
    lazy: bool = False,
) -> Path:
    """Process one JSON file and write the modified copy.

    When ``output_path`` is None, the file is rewritten into the same directory
    as the input with the original filename.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / input_path.name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if lazy and output_path.exists():
        return output_path

    with input_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, list):
        processed: Iterable[Any] = (
            _process_instance(item, transform, insert_key) for item in tqdm(data, desc=f"{input_path.name}")
        )
        data = list(processed)
    else:
        data = _process_instance(data, transform, insert_key)

    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2)

    return Path(output_path)


def process_dir(
    input_dir: Path,
    transform: Callable[[dict], dict],
    *,
    insert_key: str = "processed",
    output_dir: Path,
    lazy: bool = False,
) -> list[Path]:
    """Process all .json files in a directory, writing to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for path in input_dir.iterdir():
        if path.is_dir() or path.suffix.lower() != ".json":
            continue
        outputs.append(
            process_file(
                path,
                transform,
                insert_key=insert_key,
                output_path=output_dir / path.name,
                lazy=lazy,
            )
        )
    return outputs

#!/usr/bin/env python3
"""
Extract evaluation results from multiple JSON files into a single CSV file.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_config_from_filename(filename: str) -> dict:
    """
    Parse configuration parameters from filename.
    Example: results_eli5_LLaDA-8B-Instruct_no_wm_rrandom_s256_l256_b32_t0.0_n100.json
    Format: results_{dataset}_{model}_...
    """
    config = {}
    stem = Path(filename).stem

    # Split by underscores
    parts = stem.split("_")

    # Extract dataset (index 1 after 'results')
    if len(parts) > 1 and parts[0] == "results":
        config["dataset"] = parts[1]

    # Extract model name (index 2)
    if len(parts) > 2:
        config["model"] = parts[2]

    # Check for watermark status
    if "no_wm" in stem:
        config["watermark"] = "no"
    elif "_wm_" in stem:
        if "predict" in stem:
            config["watermark"] = "predict"
        elif "normal" in stem:
            config["watermark"] = "normal"
        elif "reverse" in stem:
            config["watermark"] = "reverse"
        elif "legacy-ahead" in stem:
            config["watermark"] = "legacy-ahead"
        elif "legacy-both" in stem:
            config["watermark"] = "legacy-both"
        else:
            config["watermark"] = "unknown"
    else:
        config["watermark"] = "unknown"

    # Parse parameters with prefixes
    for part in parts:
        # s256 -> steps/sequence length
        if match := re.match(r"^s(\d+)$", part):
            config["steps"] = int(match.group(1))
        # l256 -> length
        elif match := re.match(r"^l(\d+)$", part):
            config["length"] = int(match.group(1))
        # b32 -> block size
        elif match := re.match(r"^b(\d+)$", part):
            config["block_size"] = int(match.group(1))
        # t0.0 -> temperature
        elif match := re.match(r"^t([\d.]+)$", part):
            config["temperature"] = float(match.group(1))
        # n100 -> number of samples
        elif match := re.match(r"^n(\d+)$", part):
            config["num_samples"] = int(match.group(1))
        # rrandom -> strategy
        elif part.startswith("r") and part != "results":
            config["strategy"] = part

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation results from JSON files to CSV"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file name (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = (
        Path(args.output)
        if "/" in args.output or "\\" in args.output
        else input_dir / args.output
    )

    # Find all JSON files
    json_files = list(input_dir.glob(args.pattern))
    if not json_files:
        print(f"No JSON files found in {input_dir} matching pattern {args.pattern}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    # Collect all data
    all_rows = []

    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        try:
            with open(json_file, "r") as f:
                results = json.load(f)

            # Parse configuration from filename
            file_config = parse_config_from_filename(json_file.name)

            # Calculate averages for metrics
            detect_rates = [
                r.get("detect_rate", None)
                for r in results
                if r.get("detect_rate") is not None
            ]
            z_scores = [
                r.get("z_score", None) for r in results if r.get("z_score") is not None
            ]
            ppls = [r.get("ppl", None) for r in results if r.get("ppl") is not None]

            row = {
                "config_name": json_file.stem,
                "num_samples": len(results),
                **file_config,  # Add all parsed config parameters
                "detect_rate_avg": (
                    sum(detect_rates) / len(detect_rates) if detect_rates else ""
                ),
                "z_score_avg": sum(z_scores) / len(z_scores) if z_scores else "",
                "ppl_avg": sum(ppls) / len(ppls) if ppls else "",
            }
            all_rows.append(row)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_rows)

    # Reorder columns for better readability
    column_order = [
        "config_name",
        "dataset",
        "model",
        "watermark",
        "strategy",
        "steps",
        "length",
        "block_size",
        "temperature",
        "num_samples",
        "detect_rate_avg",
        "z_score_avg",
        "ppl_avg",
    ]

    # Only include columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    # Add any remaining columns
    column_order.extend([col for col in df.columns if col not in column_order])

    df = df[column_order]
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(all_rows)} rows to {output_path}")
    print(f"Configurations included: {', '.join(sorted(df['config_name'].unique()))}")


if __name__ == "__main__":
    main()

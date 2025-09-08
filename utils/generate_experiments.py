#!/usr/bin/env python3
"""
Generate experiment commands from a JSON configuration file.
Reads hyperparameter configurations and generates all combinations for experiments.
"""

import argparse
import json
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def generate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    # Filter out None/null values from lists but keep them as options
    filtered_params = {}
    for key, values in params.items():
        if isinstance(values, list):
            filtered_params[key] = values
        else:
            filtered_params[key] = [values]
    
    # Get parameter names and values
    param_names = list(filtered_params.keys())
    param_values = [filtered_params[name] for name in param_names]
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        combo_dict = dict(zip(param_names, combination))
        combinations.append(combo_dict)
    
    return combinations


def build_command(base_command: str, params: Dict[str, Any]) -> str:
    """Build a command line from base command and parameters."""
    cmd_parts = [base_command]
    
    # Handle watermark-related parameters
    if params.get("strategy") is None:
        # Remove watermark-related parameters if strategy is None
        watermark_params = ["ratio", "delta", "key", "prebias", "bitmap", "bitmap_dir"]
        for wp in watermark_params:
            params.pop(wp, None)
    else:
        # Auto-generate bitmap name if bitmap_dir is provided
        bitmap_dir = params.pop("bitmap_dir", None)
        if bitmap_dir and "vocab_size" in params and "ratio" in params and "key" in params:
            # Calculate bitmap name based on parameters
            vocab_size = params["vocab_size"]
            ratio = params["ratio"]
            key = params["key"]
            # Convert ratio to percentage (0.5 -> 50)
            ratio_int = int(ratio * 100)
            bitmap_name = f"bitmap_v{vocab_size}_r{ratio_int}_k{key}.bin"
            # Combine directory and filename
            if bitmap_dir != ".":
                params["bitmap"] = f"{bitmap_dir}/{bitmap_name}"
            else:
                params["bitmap"] = bitmap_name
    
    # Add parameters to command
    for key, value in params.items():
        if value is None:
            continue
        
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            # Handle list parameters
            for v in value:
                cmd_parts.append(f"--{key} {v}")
        else:
            cmd_parts.append(f"--{key} {value}")
    
    return " ".join(cmd_parts)


def generate_experiment_commands(
    config: Dict[str, Any],
    max_commands: Optional[int] = None
) -> List[str]:
    """Generate experiment commands from configuration."""
    base_command = config["base_command"]
    fixed_params = config.get("fixed_params", {})
    variable_params = config.get("variable_params", {})
    
    commands = []
    
    # Generate all combinations of variable parameters
    combinations = generate_combinations(variable_params)
    
    # Build command for each combination
    for combo in combinations:
        full_params = {**fixed_params, **combo}
        cmd = build_command(base_command, full_params)
        commands.append(cmd)
        
        if max_commands and len(commands) >= max_commands:
            break
    
    return commands


def save_bash_script(
    commands: List[str],
    output_file: str,
    description: str = ""
) -> None:
    """Save commands to a bash script file."""
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated experiment script\n")
        if description:
            f.write(f"# Description: {description}\n")
        f.write(f"# Total commands: {len(commands)}\n")
        f.write(f"# Generated at: {Path.cwd()}\n\n")
        
        # Sequential execution
        f.write("# Running experiments sequentially\n\n")
        f.write("# Create log directory\n")
        f.write("mkdir -p logs\n\n")
        
        f.write("# Initialize counters\n")
        f.write("TOTAL={}\n".format(len(commands)))
        f.write("CURRENT=0\n")
        f.write("FAILED=0\n\n")
        
        for i, cmd in enumerate(commands, 1):
            f.write(f"# Experiment {i}/{len(commands)}\n")
            f.write("CURRENT=$((CURRENT + 1))\n")
            f.write(f"echo '[{i}/{len(commands)}] Running experiment...'\n")
            f.write(f"echo 'Command: {cmd}'\n")
            f.write(f"if {cmd}; then\n")
            f.write("    echo '✓ Experiment completed successfully'\n")
            f.write("else\n")
            f.write("    echo '✗ Experiment failed'\n")
            f.write("    FAILED=$((FAILED + 1))\n")
            f.write("fi\n")
            f.write("echo '---'\n\n")
        
        f.write("# Summary\n")
        f.write("echo '==============================='\n")
        f.write("echo 'Experiment Summary:'\n")
        f.write("echo \"Total experiments: $TOTAL\"\n")
        f.write("echo \"Successful: $((TOTAL - FAILED))\"\n")
        f.write("echo \"Failed: $FAILED\"\n")
    
    # Make script executable
    Path(output_file).chmod(0o755)


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment commands from a JSON configuration file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output bash script file (default: run_<name_or_config>.sh)"
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=None,
        help="Maximum number of commands to generate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without saving to file"
    )
    
    args = parser.parse_args()
    
    # Check if file exists and is a file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    if not config_path.is_file():
        print(f"Error: '{args.config}' is not a file")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    
    # Get experiment name and description
    experiment_name = config.get("name", "")
    description = config.get("description", "")
    
    # Generate commands
    print(f"Configuration file: {args.config}")
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    if description:
        print(f"Description: {description}")
    
    commands = generate_experiment_commands(
        config,
        max_commands=args.max_commands
    )
    
    if not commands:
        print("No commands generated")
        sys.exit(1)
    
    print(f"Total combinations: {len(commands)}")
    
    # Print or save commands
    if args.dry_run:
        print("\n=== Generated Commands ===")
        for i, cmd in enumerate(commands[:10], 1):
            print(f"{i}. {cmd}")
        if len(commands) > 10:
            print(f"... and {len(commands) - 10} more commands")
    else:
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            # Use experiment name if available, otherwise use config filename
            if experiment_name:
                output_file = f"run_{experiment_name}.sh"
            else:
                config_name = Path(args.config).stem
                output_file = f"run_{config_name}.sh"
        
        # Save to file
        save_bash_script(
            commands,
            output_file,
            description=description
        )
        
        print(f"\nSaved {len(commands)} commands to: {output_file}")
        print(f"To run experiments:")
        print(f"  ./{output_file}")


if __name__ == "__main__":
    main()
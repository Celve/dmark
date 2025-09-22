#!/usr/bin/env python
"""
Collect all JSON files from multiple nested directories into a single output directory.
Supports recursive collection from multiple input directories and merges them into one output.

Duplicate handling strategies:
- rename: Add counter suffix to duplicate filenames (e.g., file_1.json, file_2.json)
- skip: Skip files with duplicate names, keeping only the first occurrence
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def find_all_json_files(root_dir: str, exclude_dirs: List[str] = None) -> List[Path]:
    """
    Recursively find all JSON files in a directory tree.
    
    Args:
        root_dir: Root directory to search
        exclude_dirs: List of directory names to exclude
    
    Returns:
        List of Path objects for all JSON files found
    """
    exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'node_modules', '.venv', 'venv']
    json_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove excluded directories from search
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        # Add JSON files from current directory
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(Path(dirpath) / filename)
    
    return json_files


def generate_output_filename(json_file: Path, root_dir: Path, preserve_structure: bool) -> str:
    """
    Generate output filename, handling duplicates by including path information.
    
    Args:
        json_file: Path to the JSON file
        root_dir: Root directory of the search
        preserve_structure: If True, preserve directory structure with underscores
    
    Returns:
        Output filename
    """
    if preserve_structure:
        # Get relative path from root and replace separators with underscores
        try:
            rel_path = json_file.relative_to(root_dir)
            # Convert path to string with underscores
            path_parts = list(rel_path.parts[:-1])  # Exclude filename
            if path_parts:
                prefix = '_'.join(path_parts)
                return f"{prefix}_{json_file.name}"
            else:
                return json_file.name
        except ValueError:
            # If not relative to root, just use filename
            return json_file.name
    else:
        return json_file.name


def collect_json_files(
    input_dirs: List[str],
    output_dir: str,
    move: bool = False,
    preserve_structure: bool = True,
    dry_run: bool = False,
    exclude_dirs: List[str] = None,
    duplicate_strategy: str = 'rename'
) -> Tuple[int, int, int]:
    """
    Collect all JSON files from multiple input directory trees to output directory.
    
    Args:
        input_dirs: List of root directories to search for JSON files
        output_dir: Directory to collect all JSON files
        move: If True, move files instead of copying
        preserve_structure: If True, preserve directory structure in filename
        dry_run: If True, only show what would be done without doing it
        exclude_dirs: List of directory names to exclude
        duplicate_strategy: How to handle duplicates - 'rename' (add counter) or 'skip'
    
    Returns:
        Tuple of (files_processed, files_skipped_error, files_skipped_duplicate)
    """
    output_path = Path(output_dir).resolve()
    
    # Create output directory if it doesn't exist
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_skipped_error = 0
    total_skipped_duplicate = 0
    duplicates = {}
    existing_files = set()  # Track all files in output directory
    
    # Process each input directory
    for input_dir in input_dirs:
        input_path = Path(input_dir).resolve()
        
        if not input_path.exists():
            print(f"Warning: Input directory '{input_dir}' not found, skipping...")
            continue
            
        if not input_path.is_dir():
            print(f"Warning: '{input_dir}' is not a directory, skipping...")
            continue
        
        print(f"\nProcessing directory: {input_dir}")
        
        # Find all JSON files in this directory
        json_files = find_all_json_files(input_path, exclude_dirs)
        
        if not json_files:
            print(f"  No JSON files found in {input_dir}")
            continue
        
        print(f"  Found {len(json_files)} JSON files")
        
        # Process each file
        processed = 0
        skipped_error = 0
        skipped_duplicate = 0
        
        for json_file in tqdm(json_files, desc=f"  Collecting from {input_path.name}"):
            # Generate output filename with input dir prefix to avoid conflicts
            if preserve_structure:
                # Include input directory name in the prefix
                input_dir_name = input_path.name
                rel_path_str = generate_output_filename(json_file, input_path, preserve_structure)
                output_name = f"{input_dir_name}_{rel_path_str}"
            else:
                output_name = json_file.name
            
            output_file = output_path / output_name
            
            # Handle duplicates
            if output_file.exists() or output_name in existing_files:
                if duplicate_strategy == 'skip':
                    # Skip duplicate files
                    skipped_duplicate += 1
                    if dry_run:
                        tqdm.write(f"    Would skip duplicate: {json_file.name}")
                    else:
                        # Show message for significant duplicates in normal mode
                        if skipped_duplicate <= 10 or skipped_duplicate % 100 == 0:
                            tqdm.write(f"    Skipped duplicate: {json_file.name}")
                    continue
                else:  # duplicate_strategy == 'rename'
                    # Add counter to filename
                    base, ext = output_name.rsplit('.', 1)
                    counter = duplicates.get(output_name, 1)
                    while True:
                        new_name = f"{base}_{counter}.{ext}"
                        new_output_file = output_path / new_name
                        if not new_output_file.exists() and new_name not in existing_files:
                            output_file = new_output_file
                            output_name = new_name
                            if dry_run:
                                tqdm.write(f"    Renaming duplicate: {json_file.name} -> {new_name}")
                            break
                        counter += 1
                    duplicates[output_name.rsplit('_', 1)[0] + '.' + ext] = counter + 1
            
            # Copy or move file
            if dry_run:
                action = "Would move" if move else "Would copy"
                print(f"  {action}: {json_file.relative_to(input_path)} -> {output_name}")
                # Track the file as "would be created" for duplicate detection
                existing_files.add(output_name)
            else:
                try:
                    if move:
                        shutil.move(str(json_file), str(output_file))
                    else:
                        shutil.copy2(str(json_file), str(output_file))
                    processed += 1
                    existing_files.add(output_name)
                except Exception as e:
                    print(f"  Error processing {json_file}: {e}")
                    skipped_error += 1
        
        total_processed += processed
        total_skipped_error += skipped_error
        total_skipped_duplicate += skipped_duplicate
        
        if not dry_run:
            summary_parts = [f"Processed: {processed} files"]
            if skipped_error > 0:
                summary_parts.append(f"Errors: {skipped_error}")
            if skipped_duplicate > 0:
                summary_parts.append(f"Duplicates: {skipped_duplicate}")
            print(f"  {', '.join(summary_parts)}")
    
    return total_processed, total_skipped_error, total_skipped_duplicate


def main():
    parser = argparse.ArgumentParser(
        description="Collect all JSON files from multiple nested directories into a single output directory"
    )
    
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more root directories to recursively search for JSON files"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Directory to collect all JSON files"
    )
    
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them"
    )
    
    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="Don't preserve directory structure in filenames (may cause more duplicates)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=['.git', '__pycache__', 'node_modules', '.venv', 'venv'],
        help="Directory names to exclude from search"
    )
    
    parser.add_argument(
        "--duplicates",
        choices=['rename', 'skip'],
        default='rename',
        help="How to handle duplicate filenames: 'rename' (add counter, default) or 'skip' (skip duplicate files)"
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    valid_input_dirs = []
    for input_dir in args.input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory '{input_dir}' not found, will skip")
        elif not os.path.isdir(input_dir):
            print(f"Warning: '{input_dir}' is not a directory, will skip")
        else:
            valid_input_dirs.append(input_dir)
    
    if not valid_input_dirs:
        print("Error: No valid input directories found")
        return
    
    # Confirm destructive operations
    if args.move and not args.dry_run:
        dirs_list = ', '.join(valid_input_dirs)
        response = input(f"This will MOVE files from: {dirs_list}\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Run collection
    print(f"{'='*60}")
    print(f"Collecting JSON files")
    print(f"From {len(valid_input_dirs)} directories:")
    for dir in valid_input_dirs:
        print(f"  - {dir}")
    print(f"To: {args.output_dir}")
    print(f"Action: {'Move' if args.move else 'Copy'}")
    print(f"Preserve structure: {not args.no_preserve_structure}")
    print(f"Duplicate handling: {args.duplicates}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}")
    
    processed, skipped_error, skipped_duplicate = collect_json_files(
        valid_input_dirs,
        args.output_dir,
        move=args.move,
        preserve_structure=not args.no_preserve_structure,
        dry_run=args.dry_run,
        exclude_dirs=args.exclude,
        duplicate_strategy=args.duplicates
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"Total files {'moved' if args.move else 'copied'}: {processed}")
    if skipped_error > 0:
        print(f"Total files skipped (errors): {skipped_error}")
    if skipped_duplicate > 0:
        print(f"Total files skipped (duplicates): {skipped_duplicate}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
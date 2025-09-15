#!/usr/bin/env python3
"""
Merge multiple CSV files with the same header into a single CSV file.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def merge_csv_files(input_dir, output_file, recursive=False):
    """
    Merge all CSV files in a directory that share the same header.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to output merged CSV file
        recursive: Whether to search subdirectories recursively
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return False
    
    # Find all CSV files
    if recursive:
        csv_files = list(input_path.rglob("*.csv"))
    else:
        csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        return False
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    # Group CSV files by header
    header_groups = {}
    
    for csv_file in csv_files:
        try:
            # Read just the header
            df_header = pd.read_csv(csv_file, nrows=0)
            header_tuple = tuple(df_header.columns.tolist())
            
            if header_tuple not in header_groups:
                header_groups[header_tuple] = []
            header_groups[header_tuple].append(csv_file)
        except Exception as e:
            print(f"Warning: Could not read '{csv_file}': {e}")
            continue
    
    if not header_groups:
        print("No valid CSV files could be read")
        return False
    
    if len(header_groups) > 1:
        print(f"Found {len(header_groups)} different header formats:")
        for i, (header, files) in enumerate(header_groups.items(), 1):
            print(f"\nGroup {i} ({len(files)} files):")
            print(f"  Header: {list(header)}")
            print(f"  Files: {[f.name for f in files[:3]]}{'...' if len(files) > 3 else ''}")
        
        # Use the group with most files by default
        largest_group = max(header_groups.items(), key=lambda x: len(x[1]))
        print(f"\nUsing the largest group with {len(largest_group[1])} files")
        selected_header, selected_files = largest_group
    else:
        selected_header, selected_files = list(header_groups.items())[0]
        print(f"All files share the same header")
    
    # Merge the selected CSV files
    print(f"\nMerging {len(selected_files)} CSV files...")
    
    dfs = []
    for csv_file in selected_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  Loaded: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  Error loading '{csv_file}': {e}")
            continue
    
    if not dfs:
        print("No data could be loaded")
        return False
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged CSV saved to: {output_path}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV files with the same header into a single CSV file"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing CSV files to merge"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged.csv",
        help="Output CSV file path (default: merged.csv)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories recursively for CSV files"
    )
    
    args = parser.parse_args()
    
    success = merge_csv_files(args.input_dir, args.output, args.recursive)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
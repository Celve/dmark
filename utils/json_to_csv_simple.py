#!/usr/bin/env python3
"""
Simple converter from threshold JSON to CSV.
Extracts only filename and threshold values.
"""

import json
import csv
import sys


def convert_json_to_csv(json_file, csv_file=None):
    """Convert JSON with threshold data to simple CSV."""
    
    # Read JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # If no output file specified, use input name with .csv extension
    if csv_file is None:
        csv_file = json_file.replace('.json', '.csv')
    
    # Open CSV for writing
    with open(csv_file, 'w', newline='') as csvfile:
        # Define headers
        fieldnames = ['filename', 'threshold_90', 'threshold_95', 'threshold_99', 'threshold_999']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Process each file result
        for file_result in data.get('per_file_results', []):
            row = {
                'filename': file_result['file'],
                'threshold_90': '',
                'threshold_95': '',
                'threshold_99': '',
                'threshold_999': ''
            }
            
            # Extract thresholds
            for threshold in file_result.get('thresholds', []):
                tpr = threshold['target_tpr']
                value = threshold['threshold']
                
                if tpr == 0.9:
                    row['threshold_90'] = value
                elif tpr == 0.95:
                    row['threshold_95'] = value
                elif tpr == 0.99:
                    row['threshold_99'] = value
                elif tpr == 0.999:
                    row['threshold_999'] = value
            
            writer.writerow(row)
    
    print(f"CSV file created: {csv_file}")
    print(f"Processed {len(data.get('per_file_results', []))} files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_csv_simple.py <input.json> [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_json_to_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
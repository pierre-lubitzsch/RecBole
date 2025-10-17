import csv
import argparse
import os
import subprocess
import sys


def deduplicate_with_external_sort(input_file, output_file, has_header=True):
    """
    Remove duplicate interactions using external sort (Unix sort command).
    This is extremely memory efficient as it uses disk-based sorting.
    
    Args:
        input_file: Input file with potential duplicates
        output_file: Output file with duplicates removed and sorted
        has_header: Whether input file has a header line to skip
    """
    
    print(f"Deduplicating and sorting {input_file} using external sort...")
    
    # Check if sort command is available
    try:
        subprocess.run(['sort', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'sort' command not found. Falling back to Python implementation.")
        return deduplicate_in_memory(input_file, output_file, has_header)
    
    # Count total records
    print("Counting total records...")
    total_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        if has_header:
            next(f)  # Skip header
        for _ in f:
            total_count += 1
    
    print(f"Total records: {total_count}")
    
    # Step 1: Sort file using Unix sort (extremely memory efficient)
    temp_sorted = 'temp_sorted_all_fields.tsv'
    temp_no_header = 'temp_no_header.tsv'
    
    print("Step 1: Sorting by all fields (for deduplication)...")
    
    # Copy data without header for sorting
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(temp_no_header, 'w', encoding='utf-8') as f_out:
        if has_header:
            next(f_in)  # Skip header
        for line in f_in:
            f_out.write(line)
    
    # Use Unix sort - sorts stably and can handle huge files with limited memory
    # Sort by all fields to group duplicates together
    sort_cmd = ['sort', '-t', '\t', '-k', '1,4', temp_no_header, '-o', temp_sorted]
    subprocess.run(sort_cmd, check=True)
    
    os.unlink(temp_no_header)
    
    # Step 2: Remove duplicates by sequential comparison (streaming - very low memory)
    print("Step 2: Removing duplicates (streaming)...")
    
    temp_unique = 'temp_unique.tsv'
    unique_count = 0
    duplicate_count = 0
    prev_line = None
    
    with open(temp_sorted, 'r', encoding='utf-8') as f_in, \
         open(temp_unique, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if line != prev_line:
                f_out.write(line)
                unique_count += 1
                prev_line = line
            else:
                duplicate_count += 1
    
    os.unlink(temp_sorted)
    
    # Step 3: Sort by user_id and timestamp for final output
    print("Step 3: Re-sorting by user_id and timestamp...")
    
    temp_final = 'temp_final.tsv'
    
    # Sort by user_id (field 1) and timestamp (field 4, numeric)
    sort_cmd = ['sort', '-t', '\t', '-k', '1,1', '-k', '4,4n', temp_unique, '-o', temp_final]
    subprocess.run(sort_cmd, check=True)
    
    os.unlink(temp_unique)
    
    # Add header and write final output
    print(f"Writing final output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        with open(temp_final, 'r', encoding='utf-8') as f_in:
            f_out.write(f_in.read())
    
    os.unlink(temp_final)
    
    print(f"\nTotal records processed: {total_count}")
    print(f"Unique records: {unique_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Duplicate rate: {duplicate_count/total_count*100:.2f}%")
    print(f"Done! Written {unique_count} unique records to {output_file}")


def deduplicate_in_memory(input_file, output_file, has_header=True):
    """
    Fallback: In-memory deduplication using streaming approach.
    Only keeps previous record in memory for comparison.
    """
    
    print(f"Deduplicating {input_file} using in-memory streaming method...")
    
    # Step 1: Sort and write to temp file
    print("Step 1: Loading and sorting...")
    records = []
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        if has_header:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                records.append(row)
                total_count += 1
        else:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                records.append({
                    'user_id': row[0],
                    'item_id': row[1],
                    'rating': row[2],
                    'timestamp': row[3]
                })
                total_count += 1
    
    print(f"Read {total_count} records, sorting...")
    records.sort(key=lambda x: (x['user_id'], x['item_id'], x['rating'], x['timestamp']))
    
    temp_sorted = 'temp_sorted.tsv'
    with open(temp_sorted, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        writer.writerows(records)
    
    del records
    
    # Step 2: Stream through and deduplicate
    print("Step 2: Streaming deduplication...")
    
    temp_dedup = 'temp_dedup.tsv'
    unique_count = 0
    duplicate_count = 0
    prev_key = None
    
    with open(temp_sorted, 'r', encoding='utf-8') as f_in, \
         open(temp_dedup, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        writer = csv.DictWriter(f_out, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        
        for row in reader:
            current_key = (row['user_id'], row['item_id'], row['rating'], row['timestamp'])
            
            if current_key != prev_key:
                writer.writerow(row)
                unique_count += 1
                prev_key = current_key
            else:
                duplicate_count += 1
    
    os.unlink(temp_sorted)
    
    # Step 3: Re-sort by user and timestamp
    print("Step 3: Re-sorting by user_id and timestamp...")
    
    records = []
    with open(temp_dedup, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        records = list(reader)
    
    os.unlink(temp_dedup)
    
    records.sort(key=lambda x: (x['user_id'], float(x['timestamp'])))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        writer = csv.DictWriter(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        writer.writerows(records)
    
    print(f"\nTotal records processed: {total_count}")
    print(f"Unique records: {unique_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Duplicate rate: {duplicate_count/total_count*100:.2f}%")
    print(f"Done! Written {unique_count} unique records to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Remove duplicate interactions from Amazon reviews dataset (memory efficient)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deduplicate the raw file (uses external sort if available)
  python deduplicate.py
  
  # Custom input/output files
  python deduplicate.py --input my_raw_data.inter --output my_clean_data.inter
  
  # Force in-memory method (if external sort fails)
  python deduplicate.py --no-external-sort

This script uses Unix 'sort' command for memory-efficient external sorting when available.
This allows processing files much larger than available RAM.
        """
    )
    parser.add_argument(
        '--input',
        default='amazon_reviews_raw.inter',
        help='Input file with potential duplicates (default: amazon_reviews_raw.inter)'
    )
    parser.add_argument(
        '--output',
        default='amazon_reviews.inter',
        help='Output file for deduplicated data (default: amazon_reviews.inter)'
    )
    parser.add_argument(
        '--no-external-sort',
        action='store_true',
        help='Force in-memory deduplication (don\'t use external sort)'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Input file has no header line (e.g., temp_unsorted.tsv)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Warn if overwriting
    if args.input == args.output:
        print("Warning: Output file is the same as input file.")
        print("The original file will be overwritten after deduplication.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    if args.no_external_sort:
        deduplicate_in_memory(args.input, args.output, has_header=not args.no_header)
    else:
        deduplicate_with_external_sort(args.input, args.output, has_header=not args.no_header)


if __name__ == "__main__":
    main()

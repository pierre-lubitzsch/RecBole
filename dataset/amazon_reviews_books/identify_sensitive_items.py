import gzip
import json
import os
import argparse
import re

def extract_sensitive_items(sensitive_keywords, output_file='sensitive_items.txt', input_files=None):
    """
    Extract item IDs for books on sensitive topics using keyword matching.
    Works with Amazon Books metadata files.

    Args:
        sensitive_keywords: List of keywords to search for (case-insensitive, whole-word matching)
        output_file: File to write sensitive item IDs to
        input_files: List of specific metadata files to process. If None, process all metadata files
    """
    sensitive_items = set()

    # Compile regex patterns for whole-word matching (case-insensitive)
    # \b ensures we match whole words only
    keyword_patterns = [
        (keyword, re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE))
        for keyword in sensitive_keywords
    ]

    print(f"Searching for books with keywords: {sensitive_keywords}")
    print("Note: This script processes METADATA files (meta_*.jsonl.gz), not review files")
    print("Note: Using whole-word matching")

    # Determine which files to process
    if input_files:
        files_to_process = input_files
        print(f"Processing specified files: {files_to_process}")
    else:
        # Process all metadata files in current directory (meta_*.jsonl.gz)
        files_to_process = [f for f in sorted(os.listdir(".")) if f.startswith("meta_") and f.endswith(".jsonl.gz")]
        if not files_to_process:
            # Fallback to any .jsonl.gz file
            files_to_process = [f for f in sorted(os.listdir(".")) if f.endswith(".jsonl.gz")]
        print(f"Processing metadata files in current directory")

    if not files_to_process:
        print("Error: No metadata files found!")
        print("Please ensure you have downloaded the metadata files (meta_*.jsonl.gz)")
        return set()

    # Process files
    for file in files_to_process:
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found, skipping...")
            continue

        if not file.endswith(".jsonl.gz"):
            print(f"Warning: File '{file}' is not a .jsonl.gz file, skipping...")
            continue

        print(f"Processing {file}...")
        item_count = 0
        processed = 0

        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed += 1

                    # Get item ID (prefer parent_asin, fallback to asin)
                    item_id = data.get('parent_asin') or data.get('asin')
                    if not item_id:
                        continue

                    # Build searchable text from various fields

                    # 1. Main category
                    main_category = data.get('main_category', '') or ''

                    # 2. Categories field (list of category paths)
                    categories = data.get('categories', [])
                    if isinstance(categories, list):
                        categories = ' '.join(str(c) for c in categories if c is not None)
                    else:
                        categories = str(categories) if categories is not None else ''

                    # 3. Title field
                    title = data.get('title', '') or ''

                    # 4. Description field
                    description = data.get('description', [])
                    if isinstance(description, list):
                        description = ' '.join(str(d) for d in description if d is not None)
                    else:
                        description = str(description) if description is not None else ''

                    # 5. Features field
                    features = data.get('features', [])
                    if isinstance(features, list):
                        features = ' '.join(str(f) for f in features if f is not None)
                    else:
                        features = str(features) if features is not None else ''

                    # 6. Details field
                    details = data.get('details', {})
                    if isinstance(details, dict):
                        details = ' '.join(str(v) for v in details.values() if v is not None)
                    else:
                        details = str(details) if details is not None else ''

                    # 7. Store name
                    store = data.get('store', '') or ''

                    # Combine all text fields
                    all_text = ' '.join([
                        main_category,
                        categories,
                        title,
                        description,
                        features,
                        details,
                        store
                    ])

                    # Check if any sensitive keyword is in the combined text (whole-word matching)
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(all_text):
                            sensitive_items.add(item_id)
                            item_count += 1
                            break

                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file}: {e}")
                    continue
                except Exception:
                    # Silently skip lines with errors
                    continue

        print(f"  Processed {processed} products, found {item_count} sensitive items in {file}")

    # Write to output file
    print(f"\nTotal unique sensitive items: {len(sensitive_items)}")

    if len(sensitive_items) == 0:
        print("\nWarning: No sensitive items found!")
        print("Possible reasons:")
        print("  1. You're processing review files instead of metadata files")
        print("  2. The keywords don't match the book descriptions")
        print("  3. The metadata files don't contain books on those topics")
        print("\nMake sure to:")
        print("  - Download metadata files (meta_*.jsonl.gz)")
        print("  - Use appropriate keywords")
        print("  - For Books dataset, use: meta_Books.jsonl.gz")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item_id in sorted(sensitive_items):
                f.write(f"{item_id}\n")

        print(f"Written sensitive item IDs to {output_file}")

    return sensitive_items

def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive books from Amazon Books metadata files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find books about mental health (default category)
  python identify_sensitive_items.py

  # Find books about specific sensitive topics
  python identify_sensitive_items.py --keywords suicide depression anxiety "mental health" --output sensitive_asins_health.txt

  # Process specific metadata files
  python identify_sensitive_items.py --files meta_Books.jsonl.gz --keywords suicide depression

  # Find books about violence
  python identify_sensitive_items.py --keywords violence murder war torture --output sensitive_asins_violence.txt

Common Sensitive Topics for Books:
  - Mental health: suicide, depression, anxiety, "mental health", "mental illness", trauma
  - Violence: violence, murder, war, torture, abuse, assault
  - Substance abuse: addiction, alcoholism, drugs, "drug abuse"
  - Sexual content: "sexual assault", rape, abuse, "sexual abuse"

Important:
  - This script requires METADATA files (meta_*.jsonl.gz), NOT review files
  - Keywords use whole-word matching (e.g., 'suicide' won't match 'suicidal')
        """
    )
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=['suicide', 'depression', 'anxiety', 'mental health', 'mental illness', 'self harm'],
        help='Keywords to search for (case-insensitive, whole-word matching)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_asins_health.txt',
        help='Output file for sensitive item IDs'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Specific metadata .jsonl.gz files to process (default: all meta_*.jsonl.gz files)'
    )

    args = parser.parse_args()

    extract_sensitive_items(args.keywords, args.output, args.files)

if __name__ == "__main__":
    main()

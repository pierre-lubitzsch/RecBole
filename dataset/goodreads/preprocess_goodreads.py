import sys
import os
import csv
import argparse


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    min_rating = args.min_rating

    print(f"Processing Goodreads interactions from {input_file}...")
    if min_rating:
        print(f"Filtering: only keeping ratings >= {min_rating}")

    record_count = 0
    filtered_count = 0

    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', newline='', encoding='utf-8') as out_f:

        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')

        for row in reader:
            try:
                user_id = row['user_id']
                item_id = row['book_id']  # Note: book_id becomes item_id
                rating = row['rating']

                # Goodreads interactions.csv doesn't have timestamp, so we use a placeholder
                # We can use the row index as a pseudo-timestamp for ordering
                timestamp = str(record_count)

                # Filter by minimum rating if specified
                if min_rating and float(rating) < min_rating:
                    filtered_count += 1
                    continue

                # Only write if all fields exist and rating > 0 (0 means no rating)
                if all([user_id, item_id, rating]) and float(rating) > 0:
                    writer.writerow({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating,
                        'timestamp': timestamp
                    })
                    record_count += 1

            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

    print(f"Total records written: {record_count}")
    if min_rating:
        print(f"Records filtered out (rating < {min_rating}): {filtered_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="goodreads_interactions.csv",
        help="Input Goodreads interactions CSV file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="goodreads_raw.inter",
        help="Output .inter file"
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=None,
        help="Minimum rating threshold (filter out ratings below this value)"
    )
    args = parser.parse_args()
    main(args)

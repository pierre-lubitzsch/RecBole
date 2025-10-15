import sys
import os
import gzip
import json
import csv


def main():
    all_records = []
    seen = set()
    
    for file in sorted(os.listdir(".")):
        if not file.endswith(".jsonl.gz"):
            continue
        
        print(f"Processing {file}...")
        
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    record = {
                        'user_id': data.get('user_id'),
                        'item_id': data.get('asin'),
                        'rating': data.get('rating'),
                        'timestamp': data.get('timestamp')
                    }
                    
                    key = (record["user_id"], record["item_id"], record["rating"], record["timestamp"])
                    if all(map(lambda x: x is not None, key)) and key not in seen:
                        all_records.append(record)
                        seen.add(key)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file}: {e}")
                    continue
    
    print(f"Total records collected: {len(all_records)}")
    
    all_records.sort(key=lambda x: (x['user_id'], x['timestamp']))
    
    output_file = 'amazon_reviews.inter'
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        writer = csv.DictWriter(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        writer.writerows(all_records)
    
    print(f"Done! Written {len(all_records)} records to {output_file}")


if __name__ == "__main__":
    main()
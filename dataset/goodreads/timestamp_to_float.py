import csv
import datetime

filename = 'goodreads.inter'  # Your inter file name
datetime_format = "%Y-%m-%d %H:%M:%S"  # Format of your timestamp, e.g., "2005-04-02 23:32:07"

# 1. Read the entire file using comma as delimiter
with open(filename, 'r', newline='') as fin:
    reader = csv.reader(fin, delimiter='\t')
    rows = list(reader)

# 2. Find the timestamp column (it should be named like "timestamp:float" in the header)
header = rows[0]
timestamp_idx = None
for i, col_name in enumerate(header):
    if col_name.startswith("timestamp:"):
        timestamp_idx = i
        break

if timestamp_idx is None:
    raise ValueError("No 'timestamp:...' column found in header.")

# 3. Convert each date/time string in the timestamp column to a numeric Unix timestamp
for row in rows[1:]:
    dt_str = row[timestamp_idx]
    dt_obj = datetime.datetime.strptime(dt_str, datetime_format)
    epoch_time = dt_obj.timestamp()
    row[timestamp_idx] = str(epoch_time)

# 4. Overwrite the original file with the updated rows, using comma as the delimiter
with open(filename, 'w', newline='') as fout:
    writer = csv.writer(fout, delimiter='\t')
    writer.writerows(rows)

print(f"Done! Overwrote {filename} with numeric timestamps using comma as a separator.")


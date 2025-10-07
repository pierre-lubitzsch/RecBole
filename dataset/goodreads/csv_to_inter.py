import csv
import datetime

input_filename = 'rating.csv'
output_filename = 'goodreads.inter'

# Define the datetime format of your timestamp strings.
datetime_format = "%Y-%m-%d %H:%M:%S"

with open(input_filename, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)  # Default delimiter is comma
    with open(output_filename, 'w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            # For the header row, set the proper atomic file header.
            if i == 0:
                row = ["userId:token", "movieId:token", "rating:float", "timestamp:float"]
            else:
                # Convert the timestamp (assumed to be at index 3) from a date string to a Unix timestamp.
                dt_str = row[3]
                dt_obj = datetime.datetime.strptime(dt_str, datetime_format)
                epoch_time = dt_obj.timestamp()
                row[3] = str(epoch_time)
            tsv_writer.writerow(row)

print(f"Conversion complete. Data saved to {output_filename}")


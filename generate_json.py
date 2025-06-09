import os
import csv
import json

# Configuration
mhd_folder = 'subset0'             # path to folder containing .mhd files
csv_path = 'annotations.csv'           # path to CSV file with seriesuid, coordX, coordY, coordZ, diameter_mm
out_json = 'testing_data.json'    # output JSON filename

# Read CSV and group boxes by seriesuid
boxes_by_series = {}
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t' if csv_path.endswith('.tsv') else ',')
    for row in reader:
        sid = row['seriesuid']
        x = float(row['coordX'])
        y = float(row['coordY'])
        z = float(row['coordZ'])
        d = float(row['diameter_mm'])
        # box format: [x, y, z, d, d, d]
        box = [x, y, z, d, d, d]
        if sid not in boxes_by_series:
            boxes_by_series[sid] = []
        boxes_by_series[sid].append(box)

# Build testing list
testing_list = []
for sid, box_list in boxes_by_series.items():
    filename = sid + '.mhd'
    # check if file exists
    file_path = os.path.join(mhd_folder, filename)
    if not os.path.exists(file_path):
        #print(f"Warning: {filename} not found in {mhd_folder}, skipping.")
        continue
    entry = {
        'box': box_list,
        'image': filename,
        'label': [0] * len(box_list)
    }
    testing_list.append(entry)

# Write out JSON
data = {'testing': testing_list}
with open(out_json, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Generated {out_json} with {len(testing_list)} entries.")

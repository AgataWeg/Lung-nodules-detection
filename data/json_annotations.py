import os
import json
import pandas as pd

# configurations
mhd_folder = "./subset0"
excel_file = "annotations.csv"
output_json = "testing_data.json"

mhd_files = [f for f in os.listdir(mhd_folder) if f.endswith(".mhd")]

df = pd.read_csv(excel_file)

data = {"testing": []}
count = 0

for file in mhd_files:
    file_id = os.path.splitext(file)[0]  # remove .mhd extension

    # find the row in the DataFrame matching the file ID
    row = df[df["seriesuid"] == file_id]

    if not row.empty:
        x, y, z, diameter = row.iloc[0][["coordX", "coordY", "coordZ", "diameter_mm"]]
        entry = {
            "box": [[float(x), float(y), float(z), diameter, diameter, diameter]],
            "image": file,
            "label": [0],
        }
        data["testing"].append(entry)
        count += 1

with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON saved to {output_json}")
print(f"Number of valid entries: {count}")

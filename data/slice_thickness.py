import os
import json
import SimpleITK as sitk

# configurations
mhd_folder = "./subset0"
output_json = "slice_thickness.json"

mhd_files = [f for f in os.listdir(mhd_folder) if f.endswith(".mhd")]

data = {"slice_thickness": []}

for file in mhd_files:
    file_id = os.path.splitext(file)[0]  # remove .mhd extension
    ds = sitk.ReadImage(mhd_folder + "/" + file)
    slice_thickness = ds.GetSpacing()[2]
    entry = {
        "slice_thickness": float(slice_thickness),
        "image": file,
    }
    data["slice_thickness"].append(entry)

# save to JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON saved to {output_json}")

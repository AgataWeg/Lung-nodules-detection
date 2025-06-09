# Instructions
Add in the data/ folder the subset0/ folder  from [*LUNA16*](https://doi.org/10.5281/zenodo.3723295).\
Add in the trained_models/ the model.ts file (download it from [monai_lung_nodule_ct_detection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection)).\
Create python environment: `python -m venv monai-det`\
Activate environment: `source monai-det/bin/activate`\
`pip install -r requirements.txt` \
Before running the script generate json files: `python generate_json.py`\
Run the script on your personal computer: `python roi_detection.py`\
Run the script on Snellius: `sbatch roi_snellius.sh`
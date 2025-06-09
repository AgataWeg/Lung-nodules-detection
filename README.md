# Lung-nodules-detection
Repository includes experimental approaches for lung nodules detection. The focus is on computed tomography (CT) and synthetic computed tomography fluoroscopy (CTF) scans. It also includes extraction of nodules sizes and locations from a 4DCT dataset, that containes a sequences of CT scans measured during patients breathing.

RetinaNet was used as the detector in this project and can be found on the NVIDIA NGC website: [monai_lung_nodule_ct_detection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection). A pre-trained model is available for download directly from that page.
To run scripts:

- create a trained_models/ directory in this workspace,

- place there trained model.


## Repository Structure
In a repository there are four directories: ROI, CTF, 4DCT, data. They use [*Luna16*](https://doi.org/10.5281/zenodo.3723295) dataset. To run the srcipts from ROI, CTF and data folders:
- download and place data from *Luna16* in a data/ directory. It is recommended to use data from *Subset0* - download and place this folder into data/ directory,
- for nodule annotations download the *annotations.csv* file and place it in data/ directory as well,
- run an appropriate script (...) to create a JSON file containing the nodule names and bounding box information.

#### data/

The folder includes a script called *slice_thickness.py* , which reads the slice thickness values from CT scans in the dataset and saves them into a single JSON file.
It also contains a script (...) responsible for creating a JSON file with scan names and corresponding nodule bounding boxes, which is later used during inference.

#### ROI/
The folder includes a script called *roi_detection.py* , which extracts the ROI from volumetric CT and performs nodule detection using RetinaNet as described in the report.\
Follow the instructions `ROI/README.md` to run the script.
#### CTF/

Contains a script that implements a pipeline for working with synthetic CTF data. It first generates a synthetic CTF scan using a specified number of CT slices. Then, it performs inference using RetinaNet, with the option to apply the region of interest either before or after inference (this can be configured by the user).

The results — including inference time, the predicted nodule bounding box, and the distance between the predicted and ground truth centers — are saved into a JSON file for further analysis.

#### 4DCT/

Includes a script that converts DICOM file into MHD and also calculates a nodule size and finds a center from a given nodule object. It was created for [*4D-Lung*](https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE) dataset. To run the srcipt:

- download and place *4D-Lung* dataset in a 4DCT/ directory.


## Sofware
The environment employed for the scripts can be found in the file [environment.txt](https://github.com/AgataWeg/Lung-nodules-detection/blob/main/environment.txt).

## Running the Scripts
#### Locally
Clone this repository to your computer. You can use both CPU or GPU.
- run: 
`git clone git@github.com:AgataWeg/Lung-nodules-detection.git`
- use [environment.txt](https://github.com/AgataWeg/Lung-nodules-detection/blob/main/environment.txt) to create an environment (e.g. using [conda](https://anaconda.org/anaconda/conda))
- follow [above points](#lung-nodules-detection) - from the beginning of this README

#### In a Server
Clone this repository to the server. Follow the same setup steps as described for the [local environment](#locally). An example shell script named *batch_.sh* is included to demonstrate how to run the pipeline in a batch mode.

## Data

[*Luna16*](https://doi.org/10.5281/zenodo.3723295)

An open dataset for lung nodule detection that uses data from the publicly available [LIDC/IDRI](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) database.

[*4D-Lung*](https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE)

A dataset consists of scans acquired during
chemoradiotherapy of 20 non-small cell lung cancer patients
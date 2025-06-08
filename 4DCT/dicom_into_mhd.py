import numpy as np
import SimpleITK as sitk
import os
import pydicom
import json

# read configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)
save_mhd_folder = config["save_mhd_folder"]
path_to_4D_lung = config["path_to_4D_lung"]

print("Output dir:", save_mhd_folder)
print("Data dir:", path_to_4D_lung)


def dicom_to_mhd(dicom_folder, output_filename_prefix):
    """
    Reads a DICOM series from the specified folder and saves it as a .mhd file
    with a corresponding .raw file.
    :param dicom_folder: Path to the folder containing DICOM files.
    :param output_filename_prefix: Prefix for the output files (without extension).
    :return: None
    """
    # read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # save as .raw and .mhd
    sitk.WriteImage(image, output_filename_prefix + ".mhd")
    print(f"Saved as {output_filename_prefix}.mhd and {output_filename_prefix}.raw")


def maximal_diameter(points):
    """
    Calculate the maximal diameter of a set of points in 3D space.
    :param points: An array of shape (N, 3) where N is the number of points.
    :return: The maximal distance between any two points in the set.
    """
    max_dist = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            max_dist = max(max_dist, dist)
    return max_dist


def calculate_diameter_and_centroid(name_dicom, target_roi_name):
    """
    Calculate the maximal diameter and centroid of a specified ROI of a nodule in a DICOM file.
    :param name_dicom: Path to the DICOM file.
    :param target_roi_name: Name of the nodule ROI to search for.
    :return: A tuple containing the maximal diameter and centroid of the ROI.
    """

    ds = pydicom.dcmread(name_dicom)

    # search for the ROI number based on the ROI name
    roi_number = None
    for roi in ds.StructureSetROISequence:
        if roi.ROIName == target_roi_name:
            roi_number = roi.ROINumber
            break

    if roi_number is not None:
        print(f"ROI number is: {roi_number}")
    else:
        print(f"There is no ROI named: {target_roi_name}")
        return None, None

    # look for contours in the ROIContourSequence and collect all points
    all_points = []
    for roi_contour in ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            for contour in roi_contour.ContourSequence:
                contour_data = contour.ContourData
                coords = np.array(contour_data).reshape(-1, 3)
                all_points.append(coords)

    if not all_points:
        print(f"No contours found for ROI number: {roi_number}")
        return None, None
    all_points = np.vstack(all_points)

    # calculate the centroid of all points
    centroid = all_points.mean(axis=0)
    # calculate the maximal diameter
    max_d = maximal_diameter(all_points)

    print("Centroid (x, y, z):", centroid)
    print("Maximal diameter:", max_d)
    return max_d, centroid


def main():
    """
    Main function to process DICOM files and calculate diameters and centroids.
    The sctructure of reading from folders and the way the centroids and diameters are calculated
    is based on the 4D-Lung dataset:
    Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall, P. J., Lu, J., & Williamson, J. F. (2016).
    Data from 4D Lung Imaging of NSCLC Patients (Version 2) [Data set].
    The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE
    """

    if not os.path.exists(save_mhd_folder):
        os.makedirs(save_mhd_folder)
    if not os.path.exists(path_to_4D_lung):
        print("4D-Lung folder does not exist. Please check the path in config.json.")
        return

    results_ = []
    for i in range(0, 20):  # for each patient
        root_folder = (
            path_to_4D_lung + "/" + str(100 + i) + "_HM10395"
        )  # patient folder
        for subdir in os.listdir(root_folder):
            subdir_path = os.path.join(root_folder, subdir)
            if os.path.isdir(subdir_path):
                dirs_with_dicom_files = [
                    d_f
                    for d_f in os.listdir(subdir_path)
                    if os.path.isdir(os.path.join(subdir_path, d_f))
                ]
                if dirs_with_dicom_files:
                    # sort directories to ensure consistent breathing phases order (0%-90%)
                    dirs_with_dicom_files.sort()
                    folder_nodule_ROI_0phase = os.path.join(
                        subdir_path,
                        dirs_with_dicom_files[
                            0
                        ],  # one folder with nodule ROI for 0% phase
                    )
                    folder_scan_0phase = os.path.join(
                        subdir_path,
                        dirs_with_dicom_files[
                            1
                        ],  # one folder with scan data for 0% phase
                    )
                    file_name = dirs_with_dicom_files[1]

                    # if the first folder does not contain a file that include nodule ROI (1-1.dcm),
                    # switch to the second one
                    if "1-1.dcm" not in os.listdir(folder_nodule_ROI_0phase):
                        folder_nodule_ROI_0phase = os.path.join(
                            subdir_path, dirs_with_dicom_files[1]
                        )
                        folder_scan_0phase = os.path.join(
                            subdir_path, dirs_with_dicom_files[0]
                        )
                        file_name = dirs_with_dicom_files[0]

                    print("Processing:", folder_nodule_ROI_0phase)
                    res = calculate_diameter_and_centroid(
                        str(folder_nodule_ROI_0phase) + "/1-1.dcm", "Tumor_c00"
                    )

                    # if nodule is not found, continue to the next patient
                    if res[0] is None:
                        continue

                    results_.append(
                        {
                            "box": [*res[1] + res[0] + res[0] + res[0]],
                            "label": [0],
                            "image": file_name + ".mhd",
                        }
                    )

                    # convert DICOM to MHD
                    dicom_to_mhd(folder_scan_0phase, save_mhd_folder + "/" + file_name)

            break  # for only the first subdir in the patient folder

    #  save results to a json file
    data = {"testing": results_}

    with open(save_mhd_folder + "/testing_4D.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()

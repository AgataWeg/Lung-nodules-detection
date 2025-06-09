import os
import time
import numpy as np
import math
import json
import torch

from utils import (
    get_image_transforms,
    get_voxel_trasform,
    get_post_transforms,
    load_detector_model,
    box_transform_roi_after,
    box_transform_roi_before,
    crop_image_roi_after,
    crop_image_roi_before,
)

# read configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

data_annotations_path = config["data_annotations_path"]
data_path = config["data_path"]
model_path = config["model_path"]
roi_size = config["roi_size"]
roi_applied_after_inference = config["roi_applied_after_inference"]
slices_num = config[
    "slices_num"
]  # number of slices in CTF which is: num_slices * 2 + 1
center_shift = config[
    "center_shift"
]  # e.g. 0 means no shift (nodule center in a middle slide), 1 means shift by 1 (nodule center one slice behind the middle slice)


# define transforms
image_transforms = get_image_transforms()
voxel_transforms = get_voxel_trasform()
post_transforms = get_post_transforms()

# load model
detector, device = load_detector_model(model_path)


def inference_CTF_pipeline(image_path, gt_box):
    """
    Create CTF by first cropping CT, the apply ROI (can be after inference as well)
    and run inference.
    :param image_path: Path to the CT scan image file.
    :param gt_box: Ground truth bounding box in the format [x, y, z, diameter_x, diameter_y, diameter_z].
    :return: A tuple containing:
        - distance: The distance between the predicted center and the ground truth center.
        - predicted_center_in_GT_area: Boolean indicating if the predicted center is within the ground truth sphere.
        - in_the_area_of_interest: Boolean indicating if the predicted center is in the ROI.
        - time_: Time taken for inference in seconds.
        - pred_nodule: The predicted nodule center coordinates and size.
    """
    # start measuring time
    start_time = time.time()

    # load the image
    data = {"image": image_path}
    transformed = image_transforms(data)
    world_ct = {
        "image": transformed["image"],
        "box": torch.tensor([gt_box]),
    }

    # determine the box of a nodule in voxel space
    voxel_bb = voxel_transforms(world_ct)
    voxel_box = voxel_bb["box"].squeeze(0).tolist()
    centers = np.round(voxel_box[:3]).astype(int).tolist()

    # select only a few slices for CTF and crop the image to the ROI (if ROI before inference)
    if roi_applied_after_inference:
        cropped_image = crop_image_roi_after(
            transformed["image"], centers, slices_num, center_shift
        )
    else:
        cropped_image = crop_image_roi_before(
            transformed["image"], centers, roi_size, slices_num, center_shift
        )

    # prepare the cropped image for inference and run inference
    with torch.no_grad():
        image_batch = cropped_image.unsqueeze(1)
        image_batch = image_batch.to(device)
        inference_outputs = detector(image_batch, use_inferer=True)
    target_box_key = inference_outputs[0][detector.target_box_key].to(torch.float32)

    print("Original CT Scan shape:", list(transformed["image"].shape))
    print(f"Cropped CT Scan shape: {list(image_batch.shape)}")
    print("Bounding Box GT: ", [round(x, 3) for x in gt_box])

    # define variables for results
    distance = np.nan
    pred_nodule = np.nan
    predicted_center_in_GT_area = 0
    in_the_area_of_interest = 0

    for i in range(len(target_box_key)):
        key_list = target_box_key[i].tolist()
        # transform the predicted nodule box to the full scan voxel space
        if roi_applied_after_inference:
            cropped_point_temp = box_transform_roi_after(
                centers, key_list, slices_num, center_shift
            )
        else:
            cropped_point_temp = box_transform_roi_before(
                centers, roi_size, key_list, slices_num, center_shift
            )

        cropped_box_temp = {
            "image": transformed["image"],
            "box": cropped_point_temp.unsqueeze(0),
        }
        # transform the cropped box to the original image space
        crop_box_temp = post_transforms(cropped_box_temp)
        pred_nodule = crop_box_temp["box"].squeeze().tolist()

        # if ROI is applied after inference, choose the center which is in the ROI
        if (not roi_applied_after_inference) or (
            (gt_box[0] - roi_size[0] // 2 < pred_nodule[0])
            and (gt_box[0] + roi_size[0] // 2 > pred_nodule[0])
            and (gt_box[1] - roi_size[1] // 2 < pred_nodule[1])
            and (gt_box[1] + roi_size[1] // 2 > pred_nodule[1])
        ):
            print("Predicted nodule: located in the ROI")
            in_the_area_of_interest = 1
            # calculate the distance between predicted and ground truth center
            distance = math.sqrt(
                (pred_nodule[0] - gt_box[0]) ** 2
                + (pred_nodule[1] - gt_box[1]) ** 2
                + (pred_nodule[2] - gt_box[2]) ** 2
            )
            print("Length between predicted and GT center in xyz space: ", distance)
            # if distance is less than the radius of the ground truth sphere
            if distance < (
                gt_box[3]  # gt_box[3] is diameter and equal to gt_box[4] and gt_box[5]
                / 2
            ):
                print("Predicted nodule: located in the GT sphere")
                predicted_center_in_GT_area = 1
            else:
                print("Predicted center: is NOT located in the GT sphere")
            break

    # end measuring time
    time_ = time.time() - start_time
    print(f"Testing time: {time_:.2f} s")

    return (
        distance,
        predicted_center_in_GT_area,
        in_the_area_of_interest,
        time_,
        pred_nodule,
    )


if __name__ == "__main__":
    """
    Main function to run the inference pipeline for CTF.
    """
    # print configuration
    print("Configuration loaded:")
    print("Data annotations path:", data_annotations_path)
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Roi size:", roi_size)
    print("ROI applied after inference:", roi_applied_after_inference)
    print("Slices number to examine:", slices_num)
    print("Center shift:", center_shift)

    # variables to count results
    count_truth = 0
    count_samples = 0
    count_in_ROI = 0

    # read data annotations
    with open(data_annotations_path, "r") as f:
        data = json.load(f)

    # lists to store results and time
    results = []
    inference_time = []

    # iterate over testing data
    for entry in data.get("testing", []):
        # get the nodule box and image name from the entry
        box = entry["box"][0]
        image_name = entry["image"]
        print(f"Image: {image_name}")

        # read name of the image and select corresponding file from the data folder
        image_path = os.path.join(
            data_path,
            image_name,
        )
        # apply inference pipeline
        res = inference_CTF_pipeline(image_path, box)
        # save results
        inference_time.append(res[3])
        count_truth += res[1]
        count_in_ROI += res[2]
        count_samples += 1

        results.append(
            {
                "box": res[4],
                "distance between GT and predicted centers": res[0],
                "inference_time": res[3],
                "image": image_name,
            }
        )

    print("Mean inference time: ", np.mean(inference_time))
    print("Max inference time: ", np.max(inference_time))

    print("Number of samples: ", count_samples)
    print("Detected in the ROI: ", count_in_ROI)
    print("Detected in the GT sphere: ", count_truth)

    # save results to a JSON file
    output_json = "results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

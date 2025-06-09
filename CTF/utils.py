import numpy as np
import torch

from monai.transforms.spatial.dictionary import ConvertBoxToPointsd
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    ConvertBoxToStandardModed,
    ConvertBoxModed,
)


def get_image_transforms():
    """
    Returns a series of transforms to process the input image.
    """
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"], pixdim=[0.703125, 0.703125, 1.25], padding_mode="border"
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=300.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


def get_voxel_trasform():
    """
    Returns a series of transforms to convert the box original coordinates
    to voxel coordinates.
    """
    return Compose(
        [
            ConvertBoxToStandardModed(box_keys=["box"], mode="cccwhd"),
            ConvertBoxToPointsd(keys=["box"]),
            AffineBoxToImageCoordinated(
                box_keys=["box"],
                box_ref_image_keys="image",
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
        ]
    )


def get_post_transforms():
    """
    Returns a series of transforms to convert the box coordinates
    from voxel space to original coordinates.
    """
    return Compose(
        [
            AffineBoxToWorldCoordinated(
                box_keys=["box"],
                box_ref_image_keys="image",
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
            ConvertBoxModed(box_keys=["box"], src_mode="xyzxyz", dst_mode="cccwhd"),
        ]
    )


def load_detector_model(model_path_=None):
    """
    Load the RetinaNet model for detection.
    :param model_path_: Path to the pre-trained model.
    :return: A RetinaNetDetector instance and the device on which the model is loaded.
    """
    base_anchor_shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    returned_layers = [1, 2]
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=base_anchor_shapes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.jit.load(model_path_).to(device)
    print(f"Load model from {model_path_}")

    detector = RetinaNetDetector(
        network=net, anchor_generator=anchor_generator, debug=False
    )
    score_thresh = 0.02
    nms_thresh = 0.22
    # set inference components
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=100,
    )
    patch_size = [192, 192, 80]
    detector.set_sliding_window_inferer(
        roi_size=patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="gaussian",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    detector.eval()
    return detector, device


def crop_image_roi_after(image, center, num_slices, shift):
    """
    Crop the image to obtain CTF that consists of a few slices that include the nodule center.
    :param image: The input image.
    :param center: The center coordinates of the nodule in voxel space.
    :param num_slices: The number of slices to include in the CTF which is (num_slices * 2 + 1).
    :param shift: The number of slices to shift the CTF middle slice from the nodule center.
    :return: The cropped image containing the CTF.
    """
    return image[
        :,
        :,
        :,
        center[2] - num_slices - shift : center[2] + num_slices + 1 - shift,
    ]


def box_transform_roi_after(center, pred, num_slices, shift):
    """
    Transform the predicted bounding box coordinates to the full image voxel space
    when ROI applied before inference.
    """
    min0 = pred[0]
    min1 = pred[1]
    min2 = center[2] - num_slices - shift + pred[2]

    max0 = pred[3]
    max1 = pred[4]
    max2 = center[2] - num_slices - shift + pred[5]

    return torch.tensor([min0, min1, min2, max0, max1, max2])


def crop_image_roi_before(image, center, roi_size, num_slices, shift):
    """
    Crop the image to obtain CTF that consists of a few slices that include the nodule center and select the ROI.
    :param image: The input image.
    :param center: The center coordinates of the nodule in voxel space.
    :param roi_size: The size of the region of interest (ROI) to be applied around the nodule center.
    :param num_slices: The number of slices to include in the CTF which is (num_slices * 2 + 1).
    :param shift: The number of slices to shift the CTF middle slice from the nodule center.
    :return: The cropped image containing the CTF with ROI applied.
    """
    return image[
        :,
        center[0] - roi_size[0] // 2 : center[0] + roi_size[0] // 2,
        center[1] - roi_size[1] // 2 : center[1] + roi_size[1] // 2,
        center[2] - num_slices - shift : center[2] + num_slices + 1 - shift,
    ]


def box_transform_roi_before(center, roi_size, pred, num_slices, shift):
    """
    Transform the predicted bounding box coordinates to the full image voxel space
    when ROI applied after inference.
    """
    min0 = center[0] - roi_size[0] // 2 + pred[0]
    min1 = center[1] - roi_size[1] // 2 + pred[1]
    min2 = center[2] - num_slices - shift + pred[2]

    max0 = center[0] - roi_size[0] // 2 + pred[3]
    max1 = center[1] - roi_size[1] // 2 + pred[4]
    max2 = center[2] - num_slices - shift + pred[5]

    return torch.tensor([min0, min1, min2, max0, max1, max2])

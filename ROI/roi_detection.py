import os
import time
import monai 
import torch
import numpy as np
import json 
from monai.transforms import (
    Compose, 
    ScaleIntensityRanged, 
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    )
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    ConvertBoxToStandardModed,
    ConvertBoxModed,
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector

def rmse(y, y_pred):
    square_diff = (y-y_pred)**2
    mse = square_diff.sum(dim=1)
    rmse = torch.sqrt(mse)
    return rmse

def voxel_center(box):
    return np.round(box[:3]).astype(int).tolist()

def crop_transform(centers, roi_size, pred):

    min0 = centers[0] - roi_size[0]//2 + pred[0]
    min1 = centers[1] - roi_size[1]//2 + pred[1]
    min2 = centers[2] - roi_size[2]//2 + pred[2]

    max0 = centers[0] - roi_size[0]//2 + pred[3]
    max1 = centers[1] - roi_size[1]//2 + pred[4]
    max2 = centers[2] - roi_size[2]//2 + pred[5]

    return torch.tensor([min0, min1, min2, max0, max1, max2])


def crop_image(image, centers, roi_size):
    return  image[
        :,  
        centers[0] - roi_size[0]//2: centers[0] + roi_size[0]//2,  
        centers[1] - roi_size[1]//2: centers[1] + roi_size[1]//2, 
        centers[2] - roi_size[2]//2: centers[2] + roi_size[2]//2,  
    ]

def point_inside_bbox(pred_point, center_point, distances):
    for p, c, d in zip(pred_point, center_point, distances):
        if p < (c - d) or p > (c + d):
            return 0
    return 1

def process_entry(entry):
    # only the first box
    load0 = time.time()
    first_box = entry['box'][0]
    image_file = entry['image']
    image_path = os.path.join(base_image_dir, image_file)
    #data = {'image': image_path, 'box': torch.tensor([first_box])}

    # load and preprocess image
    transformed = test_transforms({'image': image_path})
    world_ct = {'image': transformed['image'], 'box': torch.tensor([first_box])}
    
    # get voxel center
    voxel_bb = voxel_transforms(world_ct)
    centers = voxel_center(voxel_bb['box'].squeeze().numpy())
    print(f"Image: {image_file}, Voxel center: {centers}")
    print("Original CT Scan shape:", list(transformed["image"].shape))

    cropped_image = crop_image(transformed["image"], centers, roi_size)
    load_time = time.time() - load0

    torch.cuda.synchronize() if device.type=="cuda" else None
    t0 = time.time()
    detector.eval()
    with torch.no_grad():
        image_batch = cropped_image.unsqueeze(1).to(device)
        inference_outputs = detector(image_batch, use_inferer=False)

    target_box_key = inference_outputs[0][detector.target_box_key].to(torch.float32) 
    key_list = target_box_key[0].cpu().detach().numpy().tolist()

    cropped_point = crop_transform(centers, roi_size, key_list)

    torch.cuda.synchronize() if device.type=="cuda" else None
    t1 = time.time()
    inf_time = t1 - t0

    cropped_boxes = {
        "image": transformed["image"],
        "box": cropped_point.unsqueeze(0),
    }
    crop_box = post_transforms(cropped_boxes)

    print(f'Load time: {load_time:.4f} s')
    print(f'Inference time: {inf_time:.4f} s')
    print(f"Cropped CT Scan shape: {list(image_batch.shape)}")
    print("Bounding Box GT: ",[round(x, 3) for x in first_box])
    print("Bounding Box Pred: ", [round(x, 3) for x in crop_box["box"].squeeze().tolist()])

    # Check if nodule centers are inside gt bbox
    inside = point_inside_bbox(first_box[:3], crop_box["box"].squeeze().tolist()[:3], first_box[-3:])

    return first_box, crop_box["box"].squeeze().tolist(), inside, inf_time, load_time

test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=[0.703125, 0.703125, 1.25], padding_mode="border"),
            ScaleIntensityRanged(keys=["image"],a_min=-1024,a_max=300.0,b_min=0.0,b_max=1.0,clip=True,),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
voxel_transforms = Compose(
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
post_transforms = Compose(
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

data_json = 'testing_data.json'  
base_image_dir = 'data/subset0'
with open(data_json) as f:
    data_list = json.load(f)['testing']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

base_anchor_shapes = [[6,8,4],[8,6,5],[10,10,6]]
returned_layers = [1,2]
anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=base_anchor_shapes,
    )

model_path = os.path.join("trained_models", "model.ts")
net = torch.jit.load("trained_models/model.ts").to(device)
print(f"Load model from {model_path}")


detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False)
score_thresh = 0.02
nms_thresh = 0.22
detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=100,
    )

start_time = time.time()
i = 0
gts = []
crops = []
tp = []
inf_times = []
load_times = []
diameters = []
roi_size = [80,80,40]

for entry in data_list:
    i+=1
    print('Iter: ',i)
    gt, crop, inside, inf_time, load_time = process_entry(entry)
    gts.append(gt[:3])
    crops.append(crop[:3])
    tp.append(inside)
    inf_times.append(inf_time)
    load_times.append(load_time)
    diameters.append(gt[3])

# Results
times = np.array(inf_times)
load_times = np.array(load_times)
crop_tensor = torch.tensor(crops)
gt_tensor = torch.tensor(gts)
diameters = torch.tensor(diameters)
rmse_centers = rmse(gt_tensor, crop_tensor)
accuracy = 100*torch.tensor(tp).sum() / len(tp)
torch.save(diameters, 'diameters.pt')
torch.save(rmse_centers, 'rmse_ct.pt')

print(f'\n --------------------------------- \n')
print(f"Accuracy: {accuracy:.2f} %")
print("Loading/Preprocessing time over {} samples:".format(len(times)))
print(f"  Min:  {load_times.min():.4f} s")
print(f"  Max:  {load_times.max():.4f} s")
print(f"  Mean: {load_times.mean():.4f} s")
print(f"  Std:  {load_times.std():.4f} s")
print("Inference time over {} samples:".format(len(times)))
print(f"  Min:  {times.min():.4f} s")
print(f"  Max:  {times.max():.4f} s")
print(f"  Mean: {times.mean():.4f} s")
print(f"  Std:  {times.std():.4f} s")
print(f"Testing time: {(time.time()-start_time)/60:.2f} mins ")
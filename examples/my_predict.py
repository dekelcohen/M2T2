import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from m2t2.dataset import collate
from m2t2.m2t2 import M2T2
from m2t2.dataset_utils import sample_points, normalize_rgb


def predict_grasps(
    xyz: np.ndarray,
    rgb_image: Image.Image,
    depth: np.ndarray,
    cfg: DictConfig,
    model: M2T2
):
    """
    Predicts grasps for a given point cloud and its corresponding RGB image.

    Args:
        xyz (np.ndarray): A numpy array of shape (N, 3) representing the point cloud coordinates.
        rgb_image (Image.Image): A PIL Image object of shape (H, W).
        depth (np.ndarray): A numpy array of shape (H, W) representing the depth map.
        cfg (DictConfig): The configuration object for the model.
        model (M2T2): The M2T2 model instance.

    Returns:
        A list of predicted grasps.
    """
    assert xyz.shape[1] == 3, "xyz should have shape (N, 3)"
    assert depth.shape[0] == rgb_image.height and depth.shape[1] == rgb_image.width, \
        "RGB image and depth map must have the same dimensions"

    # --- 1. Process RGB Image ---
    # Normalize the RGB image and convert to a tensor
    rgb_tensor = normalize_rgb(rgb_image).permute(1, 2, 0) # H, W, C

    # --- 2. Align RGB data with Point Cloud ---
    # The point cloud `xyz` is generated from the depth map.
    # We need to select the RGB values that correspond to the valid points in the cloud.
    valid_points_mask = depth > 0
    rgb_for_points = rgb_tensor[valid_points_mask]

    assert xyz.shape[0] == rgb_for_points.shape[0], \
        "Mismatch between number of points in xyz and number of valid pixels. Ensure xyz was generated from the provided depth map."

    # --- 3. Prepare Model Inputs ---
    xyz_tensor = torch.from_numpy(xyz).float()
    
    # Center the point cloud
    xyz_centered = xyz_tensor - xyz_tensor.mean(dim=0)

    # Combine centered xyz with corresponding rgb values
    inputs = torch.cat([xyz_centered, rgb_for_points], dim=1)
    # For this minimal example, we assume the entire scene is the object of interest.
    obj_inputs = inputs.clone()

    data = {
        'inputs': inputs,
        'points': xyz_tensor,
        'seg': torch.zeros(xyz_tensor.shape[0], dtype=torch.int64),  # Dummy segmentation
        'object_inputs': obj_inputs,
        'task': 'pick'
    }

    # Sample points as done in the demo
    pt_idx = sample_points(data['points'], cfg.data.num_points)
    data['inputs'] = data['inputs'][pt_idx]
    data['points'] = data['points'][pt_idx]
    data['seg'] = data['seg'][pt_idx]
    pt_idx = sample_points(data['object_inputs'], cfg.data.num_object_points)
    data['object_inputs'] = data['object_inputs'][pt_idx]

    # Collate and move to GPU
    data_batch = collate([data])
    for key in data_batch:
        if isinstance(data_batch[key], torch.Tensor):
            data_batch[key] = data_batch[key].cuda()

    # --- 4. Perform Inference ---
    with torch.no_grad():
        outputs = model.infer(data_batch, cfg.eval)

    # Move outputs to cpu
    for key in outputs:
        if isinstance(outputs[key][0], list):
             for i in range(len(outputs[key][0])):
                 if isinstance(outputs[key][0][i], torch.Tensor):
                    outputs[key][0][i] = outputs[key][0][i].cpu()
        elif isinstance(outputs[key][0], torch.Tensor):
            outputs[key][0] = outputs[key][0].cpu()

    return outputs['grasps'][0]


@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    """
    Main function to load the model and run a sample prediction.
    """
    # --- 1. Load Model ---
    model = M2T2.from_config(cfg.m2t2)
    # Make sure to provide a valid path to your model checkpoint
    # For example: cfg.eval.checkpoint = "/path/to/your/checkpoint.pth"
    if not cfg.eval.checkpoint:
        print("WARNING: cfg.eval.checkpoint is not set. Using a randomly initialized model.")
        # raise ValueError("Checkpoint path is not set. Please set cfg.eval.checkpoint")
    else:
        ckpt = torch.load(cfg.eval.checkpoint)
        model.load_state_dict(ckpt['model'])

    model = model.cuda().eval()

    # --- 2. Create Sample Data ---
    # Replace this with your actual data loading process
    height, width = 480, 640
    # a. Create a dummy depth map and a corresponding point cloud
    # In a real scenario, you would load your depth map and use your camera intrinsics
    # to convert it to a point cloud.
    sample_depth = np.random.rand(height, width).astype(np.float32)
    sample_depth[100:300, 200:400] = 1.5 # Simulate an object
    sample_depth[sample_depth < 0.8] = 0 # Remove background
    
    # Create a dummy point cloud from the depth map (simplified, no intrinsics)
    # This creates an "unprojected" point cloud. Replace with your depth_to_xyz logic.
    y, x = np.where(sample_depth > 0)
    z = sample_depth[y, x]
    sample_xyz = np.stack((x, y, z), axis=-1).astype(np.float32)
    # Normalize x and y to be more realistic
    sample_xyz[:, 0] = (sample_xyz[:, 0] - width / 2) / (width / 2)
    sample_xyz[:, 1] = (sample_xyz[:, 1] - height / 2) / (height / 2)


    # b. Create a dummy PIL RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image_array[100:300, 200:400] = [255, 128, 0] # Make the object orange
    sample_rgb_image = Image.fromarray(image_array)


    # --- 3. Predict Grasps ---
    predicted_grasps = predict_grasps(sample_xyz, sample_rgb_image, sample_depth, cfg, model)

    # --- 4. Print Results ---
    print(f"Predicted {len(predicted_grasps)} grasp sets.")
    for i, grasps in enumerate(predicted_grasps):
        print(f"  Set {i}: {grasps.shape[0]} grasps")
        # Each grasp is a 4x4 transformation matrix
        # print(grasps)


if __name__ == '__main__':
    main()

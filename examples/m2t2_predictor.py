'''
Demo script that runs M2T2 and saves outputs for GraspMolmo.
'''
import hydra
import numpy as np
import torch
from PIL import Image
import os

from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu


def load_and_predict(data_dir, cfg):
    data, meta_data = load_rgb_xyz(
        data_dir, cfg.data.robot_prob,
        cfg.data.world_coord, cfg.data.jitter_scale,
        cfg.data.grid_resolution, cfg.eval.surface_range
    )
    if 'object_label' in meta_data:
        data['task'] = 'place'
    else:
        data['task'] = 'pick'

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    inputs, xyz, seg = data['inputs'], data['points'], data['seg']
    obj_inputs = data['object_inputs']
    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
        'placements': [],
        'placement_confidence': [],
        'placement_contacts': []
    }
    for _ in range(cfg.eval.num_runs):
        pt_idx = sample_points(xyz, cfg.data.num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz[pt_idx]
        data['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        data['object_inputs'] = obj_inputs[pt_idx]
        data_batch = collate([data])
        to_gpu(data_batch)

        with torch.no_grad():
            model_ouputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_ouputs)
        for key in outputs:
            if 'place' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_ouputs[key][0])
                ]
            else:
                outputs[key].extend(model_ouputs[key][0])
    data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
    data['object_inputs'] = obj_inputs
    return data, outputs, meta_data


@hydra.main(config_path='..', config_name='config', version_base='1.3')
def main(cfg):
    data, outputs, meta_data = load_and_predict(cfg.eval.data_dir, cfg)

    output_dir = "M2T2_grasp_outputs"
    os.makedirs(output_dir, exist_ok=True)

    rgb = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb_img_np = (rgb.numpy() * 255).astype('uint8')
    rgb_img = Image.fromarray(rgb_img_np)
    rgb_img.save(os.path.join(output_dir, "rgb_image.png"))

    xyz = data['points'].numpy()
    np.save(os.path.join(output_dir, "point_cloud.npy"), xyz)

    cam_pose = data['cam_pose'].double().numpy()
    np.save(os.path.join(output_dir, "camera_pose.npy"), cam_pose)
    
    cam_K = meta_data['intrinsics']
    np.save(os.path.join(output_dir, "camera_intrinsics.npy"), cam_K)

    if data['task'] == 'pick':
        all_grasps = []
        for grasps in outputs['grasps']:
            all_grasps.append(grasps)
        all_grasps = torch.cat(all_grasps, dim=0).numpy()

        if not cfg.eval.world_coord:
            # Grasps are in camera frame, which is what GraspMolmo expects
            grasps_for_molmo = all_grasps
        else:
            # Convert grasps from world to camera frame for GraspMolmo
            cam_pose_inv = np.linalg.inv(cam_pose)
            grasps_for_molmo = cam_pose_inv @ all_grasps
        
        np.save(os.path.join(output_dir, "grasps.npy"), grasps_for_molmo)
        print(f"Saved {len(grasps_for_molmo)} grasps and other data to {output_dir}")
    else:
        print("Task is not 'pick', no grasps saved.")


if __name__ == '__main__':
    main()

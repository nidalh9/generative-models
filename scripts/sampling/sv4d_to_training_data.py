import os
import sys
from glob import glob
from typing import List, Optional
import json
import shutil
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from fire import Fire
from PIL import Image
from scripts.demo.sv4d_helpers import (
    load_model,
    preprocess_video,
    read_video,
    run_img2vid,
    save_video,
)
from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder

sv4d2_configs = {
    "sv4d2": {
        "T": 12,  # number of frames per sample
        "V": 4,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
    "sv4d2_8views": {
        "T": 5,  # number of frames per sample
        "V": 8,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2_8views.yaml",
        "version_dict": {
            "T": 5 * 8,
            "options": {
                "discretization": 1,
                "cfg": 2.5,
                "min_cfg": 1.5,
                "num_views": 8,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 5,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}


def save_frame_as_image(frame_tensor, output_path):
    """Save a frame tensor as an image file."""
    # Convert from [-1, 1] to [0, 255]
    frame_np = ((frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    Image.fromarray(frame_np).save(output_path)


def convert_sv4d_to_blender_matrix(azimuth_rad, polar_rad, distance=4.0):
    """
    Convert SV4D camera parameters (azimuth and polar angles) to Blender-style 4x4 transformation matrix.
    
    Args:
        azimuth_rad: Azimuth angle in radians
        polar_rad: Polar angle in radians (from z-axis)
        distance: Distance from origin to camera
    
    Returns:
        4x4 transformation matrix in Blender coordinate system
    """
    # Convert from spherical to Cartesian coordinates
    # SV4D uses polar angle from z-axis
    x = distance * np.sin(polar_rad) * np.cos(azimuth_rad)
    y = distance * np.sin(polar_rad) * np.sin(azimuth_rad)
    z = distance * np.cos(polar_rad)
    
    # Camera position
    camera_pos = np.array([x, y, z])
    
    # Look-at vector (pointing towards origin)
    look_at = -camera_pos / np.linalg.norm(camera_pos)
    
    # Up vector (approximate, will be adjusted)
    up = np.array([0, 0, 1])
    
    # Compute right vector
    right = np.cross(look_at, up)
    right = right / np.linalg.norm(right)
    
    # Recompute up vector to ensure orthogonality
    up = np.cross(right, look_at)
    up = up / np.linalg.norm(up)
    
    # Create rotation matrix (camera to world)
    # In Blender convention:
    # - Column 0: Right vector
    # - Column 1: Up vector  
    # - Column 2: -Forward vector (look_at)
    # - Column 3: Translation
    transform_matrix = np.eye(4)
    transform_matrix[:3, 0] = right
    transform_matrix[:3, 1] = up
    transform_matrix[:3, 2] = -look_at  # Negative because camera looks in -Z direction
    transform_matrix[:3, 3] = camera_pos
    
    return transform_matrix.tolist()


def generate_transforms_json(n_frames, n_views, azimuths_rad, polars_rad, output_path, camera_angle_x=0.6911112070083618):
    """
    Generate transforms_train.json file with camera parameters for all frames and views.
    
    Args:
        n_frames: Number of temporal frames
        n_views: Number of views
        azimuths_rad: Array of azimuth angles in radians for each view
        polars_rad: Array of polar angles in radians for each view
        output_path: Path to save the JSON file
        camera_angle_x: Camera FOV angle in radians (default from 2MoreBalls dataset)
    """
    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    # Generate normalized time values for each frame
    time_values = np.linspace(0, 1, n_frames)
    
    frame_idx = 0
    for t in range(n_frames):
        for v in range(n_views):
            # Create frame entry
            frame_entry = {
                "file_path": f"./train/r_{frame_idx:03d}",
                "rotation": 0.0,  # Not used in most NeRF implementations
                "time": float(time_values[t]),
                "transform_matrix": convert_sv4d_to_blender_matrix(
                    azimuths_rad[v],
                    polars_rad[v],
                    distance=4.0  # Standard distance used in most datasets
                )
            }
            transforms_data["frames"].append(frame_entry)
            frame_idx += 1
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump(transforms_data, f, indent=4)
    
    return transforms_data


def create_training_dataset(
    img_matrix, 
    output_folder, 
    dataset_name,
    azimuths_rad,
    polars_rad,
    n_views,
    camera_angle_x=0.6911112070083618
):
    """
    Create a training dataset folder structure similar to 2MoreBalls with generated frames.
    
    Args:
        img_matrix: Matrix of generated images [n_frames][n_views]
        output_folder: Base output folder
        dataset_name: Name of the dataset folder
        azimuths_rad: Azimuth angles in radians
        polars_rad: Polar angles in radians
        n_views: Number of views
        camera_angle_x: Camera FOV angle
    
    Returns:
        Path to created dataset folder
    """
    # Create dataset folder structure
    dataset_path = os.path.join(output_folder, dataset_name)
    train_path = os.path.join(dataset_path, "train")
    os.makedirs(train_path, exist_ok=True)
    
    n_frames = len(img_matrix)
    
    # Save all frames as images
    frame_idx = 0
    for t in range(n_frames):
        for v in range(n_views):
            if img_matrix[t][v] is not None:
                frame_path = os.path.join(train_path, f"r_{frame_idx:03d}.png")
                save_frame_as_image(img_matrix[t][v], frame_path)
            frame_idx += 1
    
    # Generate and save transforms_train.json
    transforms_path = os.path.join(dataset_path, "transforms_train.json")
    generate_transforms_json(
        n_frames=n_frames,
        n_views=n_views,
        azimuths_rad=azimuths_rad,
        polars_rad=polars_rad,
        output_path=transforms_path,
        camera_angle_x=camera_angle_x
    )
    
    print(f"Training dataset created at: {dataset_path}")
    print(f"  - Total frames: {frame_idx}")
    print(f"  - Frames per timestep: {n_views}")
    print(f"  - Number of timesteps: {n_frames}")
    
    return dataset_path


def sample_with_training_output(
    input_path: str = "assets/sv4d_videos/camel.gif",
    model_path: Optional[str] = "checkpoints/sv4d2.safetensors",
    output_folder: Optional[str] = "outputs/training_data",
    dataset_name: Optional[str] = None,
    num_steps: Optional[int] = 50,
    img_size: int = 576,
    n_frames: int = 21,
    seed: int = 23,
    encoding_t: int = 8,
    decoding_t: int = 4,
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.9,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
    camera_angle_x: float = 0.6911112070083618,
    camera_distance: float = 4.0,
):
    """
    Generate multiple novel-view videos using SV4D and create training-ready dataset.
    
    This function extends simple_video_sample_4d2.py to also generate:
    - Organized folder structure for training (similar to dnerf/2MoreBalls)
    - transforms_train.json with Blender-format camera matrices
    - Individual frame images for training
    
    Args:
        input_path: Path to input video or image folder
        model_path: Path to SV4D model checkpoint
        output_folder: Base folder for training data output
        dataset_name: Name for the dataset folder (auto-generated if None)
        num_steps: Number of diffusion steps
        img_size: Image resolution
        n_frames: Number of frames to generate
        seed: Random seed
        encoding_t: Number of frames encoded at a time
        decoding_t: Number of frames decoded at a time
        device: Device to run on (cuda/cpu)
        elevations_deg: Elevation angles in degrees
        azimuths_deg: Azimuth angles in degrees
        image_frame_ratio: Ratio for image framing
        verbose: Verbose output
        remove_bg: Whether to remove background
        camera_angle_x: Camera FOV angle in radians for transforms.json
        camera_distance: Distance from camera to object center
    """
    # Set model config
    assert os.path.basename(model_path) in [
        "sv4d2.safetensors",
        "sv4d2_8views.safetensors",
    ], f"Unknown model: {os.path.basename(model_path)}"
    
    sv4d2_model = os.path.splitext(os.path.basename(model_path))[0]
    config = sv4d2_configs[sv4d2_model]
    print(f"Using model: {sv4d2_model}")
    print(f"Config: {config}")
    
    T = config["T"]
    V = config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"]
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + V novel views)
    subsampled_views = np.arange(n_views)
    version_dict["H"] = H
    version_dict["W"] = W
    version_dict["C"] = C
    version_dict["f"] = F
    version_dict["options"]["num_steps"] = num_steps

    torch.manual_seed(seed)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate dataset name if not provided
    if dataset_name is None:
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        dataset_name = f"{input_basename}_sv4d_{sv4d2_model}"
    
    # Read input video frames
    print(f"Reading input: {input_path}")
    base_count = 0  # We'll handle our own file naming
    
    # Preprocess video
    temp_output = os.path.join(output_folder, "temp")
    os.makedirs(temp_output, exist_ok=True)
    
    processed_input_path = preprocess_video(
        input_path,
        remove_bg=remove_bg,
        n_frames=n_frames,
        W=W,
        H=H,
        output_folder=temp_output,
        image_frame_ratio=image_frame_ratio,
        base_count=base_count,
    )
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
    images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views
    assert (
        len(elevations_deg) == n_views
    ), f"Please provide 1 value, or a list of {n_views} values for elevations_deg! Given {len(elevations_deg)}"
    
    if azimuths_deg is None:
        azimuths_deg = (
            np.array([0, 60, 120, 180, 240])
            if sv4d2_model == "sv4d2"
            else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
        )
    assert (
        len(azimuths_deg) == n_views
    ), f"Please provide a list of {n_views} values for azimuths_deg! Given {len(azimuths_deg)}"
    
    # Convert to radians
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )
    
    # Store absolute azimuths for transforms.json (in Blender coordinate system)
    absolute_azimuths_rad = np.array([np.deg2rad(a) for a in azimuths_deg])

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    # Load SV4D model
    print("Loading SV4D model...")
    model, _ = load_model(
        model_config,
        device,
        version_dict["T"],
        num_steps,
        verbose,
        model_path,
    )
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t

    # Sampling novel-view videos
    print("Generating novel views...")
    v0 = 0
    view_indices = np.arange(V) + 1
    t0_list = (
        range(0, n_frames, T-1)
        if sv4d2_model == "sv4d2"
        else range(0, n_frames - T + 1, T - 1)
    )
    
    for t0 in tqdm(t0_list, desc="Processing temporal chunks"):
        if t0 + T > n_frames:
            t0 = n_frames - T
        frame_indices = t0 + np.arange(T)
        print(f"  Sampling frames {frame_indices}")
        
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        
        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
        
        cond_mv = False if t0 == 0 else True
        samples = run_img2vid(
            version_dict,
            model,
            image,
            seed,
            polars,
            azims,
            cond_motion,
            cond_view,
            decoding_t,
            cond_mv=cond_mv,
        )
        samples = samples.view(T, V, 3, H, W)

        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    # Create training dataset with proper folder structure and transforms.json
    print("\nCreating training dataset...")
    dataset_path = create_training_dataset(
        img_matrix=img_matrix,
        output_folder=output_folder,
        dataset_name=dataset_name,
        azimuths_rad=absolute_azimuths_rad,
        polars_rad=polars_rad,
        n_views=n_views,
        camera_angle_x=camera_angle_x
    )
    
    # Also save videos for visualization (optional)
    videos_path = os.path.join(dataset_path, "videos")
    os.makedirs(videos_path, exist_ok=True)
    
    for v in range(n_views):
        vid_file = os.path.join(videos_path, f"view_{v:02d}.mp4")
        print(f"Saving video: {vid_file}")
        save_video(
            vid_file,
            [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None],
        )
    
    # Clean up temporary files
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    
    print(f"\n✅ Training dataset successfully created at: {dataset_path}")
    print(f"   - transforms_train.json generated with Blender camera format")
    print(f"   - {n_frames * n_views} frames saved in train/ folder")
    print(f"   - Videos saved in videos/ folder for visualization")
    
    return dataset_path


if __name__ == "__main__":
    Fire(sample_with_training_output)
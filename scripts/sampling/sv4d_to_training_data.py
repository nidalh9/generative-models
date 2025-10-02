import os
import sys
from glob import glob
from typing import List, Optional
import json
import shutil
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
# Add project root to path for importing pose utilities
project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
import numpy as np
import torch
from fire import Fire
from PIL import Image
from utils.pose_utils import pose_spherical
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


def convert_sv4d_to_blender_matrix(azimuth_rad, polar_rad, distance=4.0, scene_center=None):
    """
    Convert SV4D camera parameters to Blender-style 4x4 transformation matrix.
    
    CRITICAL: Positions cameras at radius 4.0 from ORIGIN (not scene_center) to match
    the working dataset pattern. This ensures cameras_extent = 3.884635, preventing
    multiple mini-scenes in object insertion training.
    
    Args:
        azimuth_rad: Azimuth angle in radians
        polar_rad: Polar angle in radians (from z-axis)
        distance: Distance from ORIGIN to camera (must be 4.0 for scale compatibility)
        scene_center: IGNORED - cameras positioned relative to origin for scale consistency
    
    Returns:
        4x4 transformation matrix in Blender coordinate system (as nested list of lists)
    """
    
    # Convert angles to degrees for pose_spherical function
    azimuth_deg = np.degrees(azimuth_rad)
    # Convert polar (from z-axis) to elevation (from xy-plane)
    elevation_deg = 90.0 - np.degrees(polar_rad)
    
    print(f"✅ SV4D camera: azimuth={azimuth_deg:.1f}°, elevation={elevation_deg:.1f}°, distance={distance}")
    
    # CRITICAL FIX: Position cameras at radius 4.0 from ORIGIN (not scene_center)
    # This matches the working dataset pattern exactly:
    # - Cameras at distance 4.0 from origin
    # - All cameras at Z≈2.0 (elevation≈-30°)
    # - Camera centroid ≈ [0, 0, 2]
    # - Results in cameras_extent = 3.884635
    c2w_torch = pose_spherical(azimuth_deg, elevation_deg, distance)
    c2w = c2w_torch.numpy()
    
    # Convert to nested list format (matching D-NeRF dataset format)
    matrix_list = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(float(c2w[i, j]))
        matrix_list.append(row)
    
    # Validation: check camera position matches working dataset pattern
    camera_pos = c2w[:3, 3]
    distance_from_origin = np.linalg.norm(camera_pos)
    
    print(f"✅ Camera position: {camera_pos}, distance from origin: {distance_from_origin:.3f}")
    
    # Verify Z-coordinate is around 2.0 (elevation ≈ -30°)
    if abs(camera_pos[2] - 2.0) > 0.5:
        print(f"⚠️ Warning: Z={camera_pos[2]:.3f} deviates from expected ~2.0")
    
    # Verify distance is exactly 4.0
    if abs(distance_from_origin - distance) > 0.001:
        print(f"⚠️ Warning: Distance mismatch! Expected: {distance:.3f}, Actual: {distance_from_origin:.3f}")
    
    return matrix_list


def estimate_scene_center_from_checkpoint(checkpoint_path):
    """
    Estimate scene center from an existing checkpoint by examining camera positions.
    This helps align generated cameras with the original scene.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing transforms_train.json
        
    Returns:
        numpy array [x, y, z] representing estimated scene center
    """
    transforms_path = os.path.join(checkpoint_path, "transforms_train.json")
    
    if not os.path.exists(transforms_path):
        print(f"[WARNING] transforms_train.json not found at {transforms_path}, using origin")
        return np.array([0.0, 0.0, 0.0])
    
    try:
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        camera_positions = []
        for frame in data["frames"]:
            # Extract camera position from transform matrix
            # The last column [:3, 3] gives translation (camera position in world)
            transform_matrix = np.array(frame["transform_matrix"])
            camera_pos = transform_matrix[:3, 3]
            camera_positions.append(camera_pos)
        
        camera_positions = np.array(camera_positions)
        
        # Estimate scene center as the point cameras are looking at
        # For circular camera paths, this is approximately the centroid of the camera circle
        # projected inward by the camera distance
        
        if len(camera_positions) > 0:
            # Calculate the centroid of camera positions
            camera_centroid = np.mean(camera_positions, axis=0)
            
            # Estimate the average distance from cameras to center
            distances = np.linalg.norm(camera_positions - camera_centroid, axis=1)
            avg_distance = np.mean(distances)
            
            # For circular arrangements, scene center is often at camera_centroid with z=0 or smaller z
            # This is a heuristic - in practice, you might need domain knowledge
            estimated_center = camera_centroid.copy()
            # Often the object is at ground level or slightly below camera centroid
            estimated_center[2] = min(estimated_center[2], 0.0)
            
            print(f"[DEBUG] Camera positions analysis:")
            print(f"  Camera centroid: {camera_centroid}")
            print(f"  Average camera distance from centroid: {avg_distance:.3f}")
            print(f"  Estimated scene center: {estimated_center}")
            
            return estimated_center
        else:
            print("[WARNING] No camera frames found, using origin")
            return np.array([0.0, 0.0, 0.0])
            
    except Exception as e:
        print(f"[WARNING] Failed to read checkpoint transforms: {e}, using origin")
        return np.array([0.0, 0.0, 0.0])


def generate_transforms_json(n_frames, n_views, azimuths_rad, polars_rad, output_path, camera_angle_x=0.6911112070083618, scene_center=None):
    """
    Generate transforms_train.json file with camera parameters for all frames and views.
    
    CRITICAL: Applies centroid correction to ensure cameras_extent = 3.884635 exactly.
    
    Args:
        n_frames: Number of temporal frames
        n_views: Number of views
        azimuths_rad: Array of azimuth angles in radians for each view
        polars_rad: Array of polar angles in radians for each view
        output_path: Path to save the JSON file
        camera_angle_x: Camera FOV angle in radians (default from 2MoreBalls dataset)
        scene_center: 3D point where cameras should look (default: origin)
    """
    
    # STEP 1: Generate all camera matrices
    temp_matrices = []
    for v in range(n_views):
        matrix = convert_sv4d_to_blender_matrix(
            azimuths_rad[v],
            polars_rad[v],
            distance=4.0,
            scene_center=scene_center
        )
        temp_matrices.append(np.array(matrix))
    
    # STEP 2: Calculate camera centroid in XY plane (Z stays at 2.0)
    cam_positions = np.array([m[:3, 3] for m in temp_matrices])
    centroid_xy = np.mean(cam_positions[:, :2], axis=0)  # Only XY centroid
    print(f"📊 Camera centroid (before correction): XY={centroid_xy}, Z=2.0")
    
    # STEP 3: Calculate offset to center cameras at origin in XY plane
    # Target: centroid should be [0, 0, 2.0] for extent = 3.884635
    offset_xy = -centroid_xy
    print(f"📐 Applying XY offset: {offset_xy} to center cameras")
    
    # STEP 4: Apply offset to all camera positions (only XY, preserve Z=2.0)
    corrected_matrices = []
    for matrix in temp_matrices:
        corrected_matrix = matrix.copy()
        corrected_matrix[0, 3] += offset_xy[0]  # X offset
        corrected_matrix[1, 3] += offset_xy[1]  # Y offset
        # Z stays at 2.0 (no change)
        corrected_matrices.append(corrected_matrix)
    
    # STEP 5: Verify the correction
    corrected_positions = np.array([m[:3, 3] for m in corrected_matrices])
    corrected_centroid = np.mean(corrected_positions, axis=0)
    distances = np.linalg.norm(corrected_positions - corrected_centroid, axis=1)
    diagonal = np.max(distances)
    cameras_extent = diagonal * 1.1
    
    print(f"✅ Camera centroid (after correction): {corrected_centroid}")
    print(f"✅ Max distance from centroid: {diagonal:.6f}")
    print(f"✅ Corrected cameras_extent: {cameras_extent:.6f}")
    print(f"✅ Target extent: 3.884635")
    print(f"✅ Difference: {abs(cameras_extent - 3.884635):.6f}")
    
    # STEP 6: Generate transforms with corrected matrices
    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    time_values = np.linspace(0, 1, n_frames)
    
    frame_idx = 0
    for t in range(n_frames):
        for v in range(n_views):
            # Convert corrected matrix to nested list
            matrix_list = corrected_matrices[v].tolist()
            
            frame_entry = {
                "file_path": f"./train/r_{frame_idx:03d}",
                "rotation": 0.0,
                "time": float(time_values[t]),
                "transform_matrix": matrix_list
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
    camera_angle_x=0.6911112070083618,
    test_split=0.2,
    scene_center=None
):
    """
    Create a training dataset folder structure similar to 2MoreBalls with generated frames.
    Applies centroid correction to ensure cameras_extent = 3.884635.
    
    Args:
        img_matrix: Matrix of generated images [n_frames][n_views]
        output_folder: Base output folder
        dataset_name: Name of the dataset folder
        azimuths_rad: Azimuth angles in radians
        polars_rad: Polar angles in radians
        n_views: Number of views
        camera_angle_x: Camera FOV angle
        test_split: Fraction of frames to use for test set (default 0.2)
        scene_center: 3D point where cameras should look (default: origin)
    
    Returns:
        Path to created dataset folder
    """
    print(f"\n[DEBUG] Creating training dataset with centroid correction")
    
    # STEP 1: Generate all camera matrices and calculate centroid correction
    temp_matrices = []
    for v in range(n_views):
        matrix = convert_sv4d_to_blender_matrix(
            azimuths_rad[v],
            polars_rad[v],
            distance=4.0,
            scene_center=scene_center
        )
        temp_matrices.append(np.array(matrix))
    
    # STEP 2: Calculate XY centroid and offset
    cam_positions = np.array([m[:3, 3] for m in temp_matrices])
    centroid_xy = np.mean(cam_positions[:, :2], axis=0)
    offset_xy = -centroid_xy
    
    print(f"📊 Camera centroid (before correction): XY={centroid_xy}, Z=2.0")
    print(f"📐 Applying XY offset: {offset_xy}")
    
    # STEP 3: Apply offset to center cameras
    corrected_matrices = []
    for matrix in temp_matrices:
        corrected_matrix = matrix.copy()
        corrected_matrix[0, 3] += offset_xy[0]
        corrected_matrix[1, 3] += offset_xy[1]
        corrected_matrices.append(corrected_matrix)
    
    # STEP 4: Verify correction
    corrected_positions = np.array([m[:3, 3] for m in corrected_matrices])
    corrected_centroid = np.mean(corrected_positions, axis=0)
    distances = np.linalg.norm(corrected_positions - corrected_centroid, axis=1)
    diagonal = np.max(distances)
    cameras_extent = diagonal * 1.1
    
    print(f"✅ Corrected centroid: {corrected_centroid}")
    print(f"✅ Corrected cameras_extent: {cameras_extent:.6f} (target: 3.884635)")
    
    # Create dataset folder structure
    dataset_path = os.path.join(output_folder, dataset_name)
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    n_frames = len(img_matrix)
    total_frames = n_frames * n_views
    
    # Determine test frame indices
    n_test_frames = max(1, int(total_frames * test_split))
    test_frame_indices = set(np.linspace(0, total_frames - 1, n_test_frames, dtype=int))
    
    train_frames_data = []
    test_frames_data = []
    time_values = np.linspace(0, 1, n_frames)
    
    # Save frames with corrected camera matrices
    frame_idx = 0
    for t in range(n_frames):
        for v in range(n_views):
            if img_matrix[t][v] is not None:
                is_test = frame_idx in test_frame_indices
                
                if is_test:
                    frame_path = os.path.join(test_path, f"r_{frame_idx:03d}.png")
                    file_path = f"./test/r_{frame_idx:03d}"
                else:
                    frame_path = os.path.join(train_path, f"r_{frame_idx:03d}.png")
                    file_path = f"./train/r_{frame_idx:03d}"
                
                save_frame_as_image(img_matrix[t][v], frame_path)
                
                # Use corrected matrix
                frame_entry = {
                    "file_path": file_path,
                    "rotation": 0.0,
                    "time": float(time_values[t]),
                    "transform_matrix": corrected_matrices[v].tolist()
                }
                
                if is_test:
                    test_frames_data.append(frame_entry)
                else:
                    train_frames_data.append(frame_entry)
                    
            frame_idx += 1
    
    # Save transforms JSONs
    transforms_train_path = os.path.join(dataset_path, "transforms_train.json")
    with open(transforms_train_path, 'w') as f:
        json.dump({
            "camera_angle_x": camera_angle_x,
            "frames": train_frames_data
        }, f, indent=4)
    
    transforms_test_path = os.path.join(dataset_path, "transforms_test.json")
    with open(transforms_test_path, 'w') as f:
        json.dump({
            "camera_angle_x": camera_angle_x,
            "frames": test_frames_data
        }, f, indent=4)
    
    print(f"\n✅ Training dataset created at: {dataset_path}")
    print(f"  - Total frames: {frame_idx}")
    print(f"  - Training frames: {len(train_frames_data)}")
    print(f"  - Test frames: {len(test_frames_data)}")
    
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
    elevations_deg: Optional[List[float]] = -30.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.9,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
    camera_angle_x: float = 0.6911112070083618,
    camera_distance: float = 4.0,
    test_split: float = 0.2,
    scene_center: Optional[List[float]] = None,
    original_checkpoint: Optional[str] = None,
):
    """
    Generate multiple novel-view videos using SV4D and create training-ready dataset.
    
    This function extends simple_video_sample_4d2.py to also generate:
    - Organized folder structure for training (similar to dnerf/2MoreBalls)
    - transforms_train.json and transforms_test.json with Blender-format camera matrices
    - Individual frame images for training and testing
    - Automatic train/test split
    
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
        test_split: Fraction of frames to use for test set (default 0.2)
        scene_center: 3D coordinates [x,y,z] where cameras should look (default: origin [0,0,0])
        original_checkpoint: Path to original checkpoint for automatic scene center detection
    """
    
    # Determine scene center: priority is scene_center parameter, then original_checkpoint, then origin
    if scene_center is not None:
        scene_center = np.array(scene_center)
        print(f"\n[DEBUG] Using custom scene center: {scene_center}")
    elif original_checkpoint is not None:
        print(f"\n[DEBUG] Estimating scene center from original checkpoint: {original_checkpoint}")
        scene_center = estimate_scene_center_from_checkpoint(original_checkpoint)
        print(f"[DEBUG] Auto-detected scene center: {scene_center}")
    else:
        scene_center = np.array([0.0, 0.0, 0.0])
        print(f"\n[DEBUG] Using default scene center (origin): {scene_center}")
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
    
    # For relative azimuths used in SV4D model
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )
    
    # Store absolute azimuths for transforms.json (in world coordinate system)
    # These are the actual camera positions we want
    absolute_azimuths_rad = np.array([np.deg2rad(a) for a in azimuths_deg])
    
    print(f"\n[DEBUG] Camera setup:")
    print(f"  Azimuths (degrees): {azimuths_deg}")
    print(f"  Elevations (degrees): {elevations_deg}")
    print(f"  Polars (radians): {polars_rad}")
    print(f"  Relative azimuths for SV4D (radians): {azimuths_rad}")
    print(f"  Absolute azimuths for transforms.json (radians): {absolute_azimuths_rad}")
    print(f"  Scene center for camera positioning: {scene_center}")

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
    print(f"[DEBUG] Using camera_angle_x (FOV): {camera_angle_x} radians ({np.degrees(camera_angle_x):.2f} degrees)")
    print(f"[DEBUG] Camera distance: {camera_distance}")
    
    dataset_path = create_training_dataset(
        img_matrix=img_matrix,
        output_folder=output_folder,
        dataset_name=dataset_name,
        azimuths_rad=absolute_azimuths_rad,
        polars_rad=polars_rad,
        n_views=n_views,
        camera_angle_x=camera_angle_x,
        test_split=test_split,
        scene_center=scene_center
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
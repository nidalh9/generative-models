import os
import sys
from glob import glob
from typing import List, Optional

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
import json
from fire import Fire

# Add the main project directory to path to import pose_spherical
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../")))
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


def sample(
    input_path: str = "assets/sv4d_videos/camel.gif",  # Can either be image file or folder with image files
    model_path: Optional[str] = "checkpoints/sv4d2.safetensors",
    output_folder: Optional[str] = "outputs",
    num_steps: Optional[int] = 50,
    img_size: int = 576,  # image resolution
    n_frames: int = 100,  # number of input and output video frames
    seed: int = 23,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.9,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
    save_transform_json: bool = True,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    assert os.path.basename(model_path) in [
        "sv4d2.safetensors",
        "sv4d2_8views.safetensors",
    ]
    sv4d2_model = os.path.splitext(os.path.basename(model_path))[0]
    config = sv4d2_configs[sv4d2_model]
    print(sv4d2_model, config)
    T = config["T"]
    V = config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"]
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    subsampled_views = np.arange(n_views)
    version_dict["H"] = H
    version_dict["W"] = W
    version_dict["C"] = C
    version_dict["f"] = F
    version_dict["options"]["num_steps"] = num_steps

    torch.manual_seed(seed)
    output_folder = os.path.join(output_folder, sv4d2_model)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")
    base_count = len(glob(os.path.join(output_folder, "*.mp4"))) // n_views
    processed_input_path = preprocess_video(
        input_path,
        remove_bg=remove_bg,
        n_frames=n_frames,
        W=W,
        H=H,
        output_folder=output_folder,
        image_frame_ratio=image_frame_ratio,
        base_count=base_count,
        fps=30,
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
        # azimuths_deg = np.linspace(0, 360, n_views + 1)[1:] % 360
        azimuths_deg = (
            np.array([0, 60, 120, 180, 240])
            if sv4d2_model == "sv4d2"
            else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
        )
    assert (
        len(azimuths_deg) == n_views
    ), f"Please provide a list of {n_views} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    # Define view indices early
    v0 = 0
    view_indices = np.arange(V) + 1

    # Initialize JSON structure for transforms if requested (one per view)
    transforms_data = {}
    if save_transform_json:
        for v in view_indices:
            transforms_data[v] = {
                "camera_angle_x": 0.6911112070083618,  # Default FoV value
                "frames": []
            }

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]
    print(f'Creating image matrix of N_frames({n_frames}) x N_views({n_views}) with shape {len(img_matrix)} x {len(img_matrix[0])}')
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

    # Interleaved sampling: Generate sparse anchor frames first

    # Pad frames if necessary to align with window size
    padded_n_frames = n_frames
    while (padded_n_frames - T) % (T - 4) != 0:
        padded_n_frames += 1

    # Extend img_matrix with padding (duplicate last frame)
    if padded_n_frames > n_frames:
        print(f"Padding {n_frames} frames to {padded_n_frames} frames for alignment")
        for t in range(n_frames, padded_n_frames):
            img_matrix.append([None] * n_views)
            img_matrix[t][0] = images_v0[-1]  # Pad with last reference frame

    # Generate overlapping anchor windows: [0,4,8,12,16,20], [20,24,28,32,36,40], etc.
    anchor_windows = []
    start = 0
    step = 4  # Frame spacing within each window
    overlap = 4  # Frames to overlap between windows

    max_windows = (padded_n_frames // step) + 2  # Safety limit
    window_count = 0

    while start < padded_n_frames and window_count < max_windows:
        # Generate T frames starting from start, spaced by step
        window_frames = []
        for i in range(T):
            frame_idx = start + i * step
            if frame_idx < padded_n_frames:
                window_frames.append(frame_idx)

        if len(window_frames) > 0:
            anchor_windows.append(window_frames)

        # Next window starts at last frame minus overlap
        if len(window_frames) >= overlap:
            new_start = window_frames[-overlap]
            if new_start <= start:  # Prevent infinite loop
                start = start + step  # Force progress
            else:
                start = new_start
        else:
            break
        window_count += 1
    print(f"Generated {anchor_windows} anchor windows for {padded_n_frames} frames")
    # Simplified: Generate anchor frames with overlapping windows
    for window_idx, anchor_frames in enumerate(tqdm(anchor_windows)):
        if len(anchor_frames) < T:
            # Pad short windows by repeating last frame
            last_frame = anchor_frames[-1]
            anchor_frames.extend([last_frame] * (T - len(anchor_frames)))

        t0 = anchor_frames[0]
        frame_indices = np.array(anchor_frames)
        print(f"Sampling anchor frames {frame_indices}")

        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)

        # Multi-view conditioning: use existing views at t0 (empty for first window)
        if window_idx == 0:
            cond_view = torch.zeros(V, 3, H, W).to(device)  # No conditioning for first window
            cond_mv = False
        else:
            cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
            cond_mv = True

        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)

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

        print(f'Azimuths: {azimuths_deg}, Elevations: {elevations_deg}')
        for i, t in enumerate(frame_indices):
            if t < n_frames:  # Only save within original frame range
                for j, v in enumerate(view_indices):
                    img_matrix[t][v] = samples[i, j][None] * 2 - 1

                    # Collect transform data if requested
                    if save_transform_json:
                        # Use the same pose_spherical function as render.py
                        azim = azimuths_deg[v]
                        elev = elevations_deg[v]
                        elev = -elev  # Negate to match render.py convention
                        radius = 4.0  # Default radius value to match render.py
                        # print(f'Radius: {radius}')

                        # Generate transform matrix using pose_spherical function
                        # This matches exactly how render.py generates poses
                        pose_matrix = pose_spherical(azim, elev, radius)
                        transform_matrix = pose_matrix.tolist()

                        frame_data = {
                            "file_path": f"./sv4d_output/frame_{t:05d}_view_{v:03d}",
                            "rotation": 0.0,
                            "time": float(t / (n_frames - 1)),
                            "transform_matrix": transform_matrix,
                            "azimuth": float(azim),
                            "elevation": float(elev),
                            "radius": radius,
                            "frame_index": int(t),  # Convert numpy int64 to Python int
                            "view_index": int(v)    # Convert numpy int64 to Python int
                        }
                        transforms_data[v]["frames"].append(frame_data)

    # Phase 2: Dense sampling - fill gaps between anchor frames
    print("Starting dense sampling phase to fill gaps...")

    # Find all anchor frames that were generated
    anchor_positions = []
    for t in range(n_frames):
        if img_matrix[t][view_indices[0]] is not None:  # Check first novel view
            anchor_positions.append(t)

    print(f"Found {len(anchor_positions)} anchor frames at positions: {anchor_positions[:10]}...")

    # Fill gaps using T-sized windows centered on gaps
    gap_positions = []
    for t in range(n_frames):
        if img_matrix[t][view_indices[0]] is None:  # Check if frame is missing
            gap_positions.append(t)

    print(f"Found {len(gap_positions)} gap frames to fill")

    # Group consecutive gaps and process in T-sized windows
    if len(gap_positions) > 0:
        # Process gaps in chunks
        processed_frames = set()

        for gap_start in range(0, len(gap_positions), T//2):  # Overlap windows by T/2
            gap_chunk = gap_positions[gap_start:gap_start + T//2]
            if not gap_chunk:
                continue

            # Create T-sized window centered around this gap chunk
            center_frame = gap_chunk[len(gap_chunk)//2]
            window_start = max(0, center_frame - T//2)
            window_end = min(n_frames, window_start + T)

            # Adjust if window is at the end
            if window_end - window_start < T and window_end == n_frames:
                window_start = max(0, window_end - T)

            dense_frames = list(range(window_start, window_end))

            # Pad window to exactly T frames if needed
            while len(dense_frames) < T:
                dense_frames.append(dense_frames[-1])
            dense_frames = dense_frames[:T]

            print(f"Processing dense window frames {dense_frames[0]}-{dense_frames[-1]}")

            # Check if we have gaps to fill in this window
            window_gaps = [f for f in dense_frames if f in gap_positions and f not in processed_frames]
            if not window_gaps:
                continue

            image = img_matrix[dense_frames[0]][v0]
            cond_motion = torch.cat([img_matrix[t][v0] for t in dense_frames], 0)

            # Use first frame of window for multi-view conditioning if available
            if img_matrix[dense_frames[0]][view_indices[0]] is not None:
                cond_view = torch.cat([img_matrix[dense_frames[0]][v] for v in view_indices], 0)
                cond_mv = True
            else:
                # Find nearest anchor for conditioning
                nearest_anchor = None
                min_dist = float('inf')
                for anchor_pos in anchor_positions:
                    dist = abs(anchor_pos - dense_frames[0])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_anchor = anchor_pos

                if nearest_anchor is not None:
                    cond_view = torch.cat([img_matrix[nearest_anchor][v] for v in view_indices], 0)
                    cond_mv = True
                else:
                    cond_view = torch.zeros(V, 3, H, W).to(device)
                    cond_mv = False

            polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
            azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
            polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
            azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)

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

            # Save only the gap frames in this window
            for idx, t in enumerate(dense_frames):
                if t in window_gaps:
                    for j, v in enumerate(view_indices):
                        img_matrix[t][v] = samples[idx, j][None] * 2 - 1

                        # Add transform data for new frames
                        if save_transform_json:
                            azim = azimuths_deg[v]
                            elev = elevations_deg[v]
                            elev = -elev
                            radius = 4.0

                            pose_matrix = pose_spherical(azim, elev, radius)
                            transform_matrix = pose_matrix.tolist()

                            frame_data = {
                                "file_path": f"./sv4d_output/frame_{t:05d}_view_{v:03d}",
                                "rotation": 0.0,
                                "time": float(t / (n_frames - 1)),
                                "transform_matrix": transform_matrix,
                                "azimuth": float(azim),
                                "elevation": float(elev),
                                "radius": radius,
                                "frame_index": int(t),
                                "view_index": int(v)
                            }
                            transforms_data[v]["frames"].append(frame_data)

                    processed_frames.add(t)

    print("Dense sampling phase completed!")

    # Save output videos
    for v in view_indices:
        vid_file = os.path.join(output_folder, f"{base_count:06d}_v{v:03d}.mp4")
        print(f"Saving {vid_file}")
        video_frames = [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None]
        print(f"View {v}: saving {len(video_frames)} frames to {vid_file}")
        save_video(
            vid_file,
            video_frames,
            fps=30,
        )
    
    # Save transforms JSON if requested (one per view)
    if save_transform_json:
        for v in view_indices:
            json_path = os.path.join(output_folder, f"transforms_{base_count:06d}_view_{v:03d}.json")
            with open(json_path, 'w') as f:
                json.dump(transforms_data[v], f, indent=4)
            print(f"Saved transform data for view {v} to: {json_path}")


if __name__ == "__main__":
    Fire(sample)

# SV4D to Training Data Generator

This document describes the `sv4d_to_training_data.py` script, which extends the SV4D video generation pipeline to create training-ready datasets for 4D scene reconstruction (like those used in D-NeRF/4D Gaussian Splatting).

## Overview

The script generates multiple novel-view videos using SV4D and automatically creates a training dataset with:
- Organized folder structure compatible with NeRF/4D-GS training
- `transforms_train.json` with Blender-format camera matrices
- Individual frame images saved as PNG files
- Optional video files for visualization

## Usage

Basic usage:
```bash
python sv4d_to_training_data.py --input_path assets/sv4d_videos/camel.gif --output_folder outputs/training_data
```

With custom parameters:
```bash
python sv4d_to_training_data.py \
    --input_path path/to/video.mp4 \
    --model_path checkpoints/sv4d2.safetensors \
    --output_folder outputs/training_data \
    --dataset_name my_dataset \
    --num_steps 50 \
    --remove_bg True \
    --camera_angle_x 0.6911 \
    --camera_distance 4.0
```

## Output Structure

The script creates the following folder structure:
```
output_folder/
└── dataset_name/
    ├── transforms_train.json  # Camera parameters in Blender format
    ├── train/                  # Training images
    │   ├── r_000.png          # Frame 0, View 0
    │   ├── r_001.png          # Frame 0, View 1
    │   └── ...
    └── videos/                 # Optional visualization
        ├── view_00.mp4
        ├── view_01.mp4
        └── ...
```

## Key Functions

### `sample_with_training_output()`
Main function that orchestrates the entire pipeline:
- Loads and configures the SV4D model
- Processes input video
- Generates novel views using SV4D
- Creates training dataset with proper structure
- Saves transforms.json and frame images

**Parameters:**
- `input_path`: Path to input video/gif or folder with frames
- `model_path`: Path to SV4D model checkpoint (sv4d2.safetensors or sv4d2_8views.safetensors)
- `output_folder`: Base folder for training data output
- `dataset_name`: Name for the dataset folder (auto-generated if None)
- `num_steps`: Number of diffusion steps (default: 50)
- `img_size`: Image resolution (default: 576)
- `n_frames`: Number of frames to generate (default: 21)
- `seed`: Random seed for reproducibility
- `encoding_t`: Frames encoded at a time (reduce for low VRAM)
- `decoding_t`: Frames decoded at a time (reduce for low VRAM)
- `device`: Device to run on (cuda/cpu)
- `elevations_deg`: Elevation angles in degrees
- `azimuths_deg`: Azimuth angles in degrees
- `remove_bg`: Whether to remove background using rembg
- `camera_angle_x`: Camera FOV angle in radians (default: 0.6911)
- `camera_distance`: Distance from camera to object (default: 4.0)

### `convert_sv4d_to_blender_matrix()`
Converts SV4D camera parameters to Blender-compatible 4x4 transformation matrices.

**Purpose:** SV4D uses spherical coordinates (azimuth, polar) while Blender/NeRF training expects 4x4 transformation matrices. This function performs the conversion.

**Parameters:**
- `azimuth_rad`: Azimuth angle in radians
- `polar_rad`: Polar angle in radians (from z-axis)
- `distance`: Distance from origin to camera

**Returns:** 4x4 transformation matrix as nested list

**Implementation details:**
1. Converts spherical coordinates to Cartesian position
2. Computes look-at vector pointing towards origin
3. Constructs orthonormal basis (right, up, forward vectors)
4. Assembles 4x4 matrix following Blender conventions:
   - Column 0: Right vector
   - Column 1: Up vector
   - Column 2: -Forward vector (camera looks in -Z)
   - Column 3: Translation (camera position)

### `generate_transforms_json()`
Generates the transforms_train.json file with camera parameters for all frames and views.

**Purpose:** Creates a JSON file containing camera matrices and metadata required for NeRF/4D-GS training.

**Parameters:**
- `n_frames`: Number of temporal frames
- `n_views`: Number of camera views
- `azimuths_rad`: Array of azimuth angles in radians
- `polars_rad`: Array of polar angles in radians
- `output_path`: Path to save JSON file
- `camera_angle_x`: Camera FOV angle in radians

**Output format:**
```json
{
    "camera_angle_x": 0.6911,
    "frames": [
        {
            "file_path": "./train/r_000",
            "rotation": 0.0,
            "time": 0.0,
            "transform_matrix": [[...4x4 matrix...]]
        },
        ...
    ]
}
```

**Key fields:**
- `camera_angle_x`: Horizontal field of view in radians
- `file_path`: Relative path to image file (without extension)
- `time`: Normalized timestamp [0, 1] for temporal interpolation
- `transform_matrix`: 4x4 camera-to-world transformation matrix

### `create_training_dataset()`
Creates the complete training dataset folder structure with frames and transforms.

**Purpose:** Organizes generated frames into a dataset compatible with NeRF/4D-GS training pipelines.

**Parameters:**
- `img_matrix`: 2D list of image tensors [n_frames][n_views]
- `output_folder`: Base output folder
- `dataset_name`: Name of the dataset folder
- `azimuths_rad`: Azimuth angles in radians
- `polars_rad`: Polar angles in radians
- `n_views`: Number of views
- `camera_angle_x`: Camera FOV angle

**Operations:**
1. Creates folder structure (dataset/train/)
2. Saves all frames as PNG images with sequential naming
3. Generates and saves transforms_train.json
4. Prints dataset statistics

### `save_frame_as_image()`
Utility function to save a tensor frame as a PNG image.

**Purpose:** Converts PyTorch tensor to PIL Image and saves to disk.

**Parameters:**
- `frame_tensor`: Frame tensor with values in [-1, 1]
- `output_path`: Path to save the image

**Processing:**
1. Converts from [-1, 1] to [0, 255] range
2. Permutes dimensions from CHW to HWC
3. Converts to numpy array
4. Saves as PNG using PIL

## Coordinate System Conventions

### SV4D Coordinates
- Uses spherical coordinates (azimuth, polar)
- Polar angle measured from z-axis
- Azimuth measured in xy-plane

### Blender/NeRF Coordinates
- Uses 4x4 transformation matrices
- Camera looks along -Z axis in camera space
- Y-up convention in world space

The conversion handles these differences to ensure compatibility with standard NeRF training pipelines.

## Integration with 4D Training

The output of this script can be directly used for training:

1. **For D-NeRF**: Place the generated dataset folder in `data/dnerf/` and use it like the 2MoreBalls dataset
2. **For 4D Gaussian Splatting**: Use the dataset path as input to the training script

Example training command:
```bash
python train.py --source_path outputs/training_data/my_dataset --model_path output/my_model
```

## Notes

- The script automatically handles temporal consistency across views
- Camera parameters are calibrated to match standard NeRF datasets
- Background removal is recommended for better reconstruction quality
- For low VRAM environments, reduce `encoding_t` and `decoding_t` parameters

## Differences from simple_video_sample_4d2.py

While `simple_video_sample_4d2.py` only generates videos, this script additionally:
1. Creates organized training dataset structure
2. Generates Blender-format camera matrices
3. Saves individual frames as images
4. Adds temporal information for 4D reconstruction
5. Provides camera calibration parameters

This makes the output directly usable for 4D scene reconstruction without additional preprocessing.
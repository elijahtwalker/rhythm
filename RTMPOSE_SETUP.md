# RTM Pose Setup Guide

This guide explains how to use the AIST++ dataset with RTM Pose from MMPose.

## Overview

The conversion script (`convertToRTMPose.py`) converts AIST++ data to COCO format, which is compatible with RTM Pose training and inference.

## Prerequisites

1. **Extract frames from videos** (aligned with annotations):
   ```bash
   python3 extractVideoFrames.py --split pose_train --alignWithAnnotations
   python3 extractVideoFrames.py --split pose_val --alignWithAnnotations
   python3 extractVideoFrames.py --split pose_test --alignWithAnnotations
   ```

2. **Install MMPose** (if not already installed):
   ```bash
   git clone https://github.com/open-mmlab/mmpose.git
   cd mmpose
   pip install -v -e .
   ```

## Step 1: Convert Dataset to RTM Pose Format

Convert your AIST++ dataset to RTM Pose format:

```bash
# Convert training split
python3 convertToRTMPose.py --split pose_train --outputDir rtmpose_dataset

# Convert validation split
python3 convertToRTMPose.py --split pose_val --outputDir rtmpose_dataset

# Convert test split
python3 convertToRTMPose.py --split pose_test --outputDir rtmpose_dataset
```

This creates:
```
rtmpose_dataset/
├── train/
│   ├── images/          # Frame images
│   └── annotations/
│       └── train.json   # COCO format annotations
├── val/
│   ├── images/
│   └── annotations/
│       └── val.json
└── test/
    ├── images/
    └── annotations/
        └── test.json
```

## Step 2: Configure RTM Pose

The dataset is now in COCO format. You can use it with RTM Pose configs. Example usage:

```python
# In your RTM Pose config file
dataset_type = 'CocoDataset'
data_root = 'rtmpose_dataset'
ann_file = 'train/annotations/train.json'
data_prefix = dict(img='train/images/')
```

## Step 3: Training

Use the converted dataset with RTM Pose training:

```bash
# Example training command (adjust paths as needed)
python tools/train.py \
    configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py \
    --work-dir work_dirs/rtmpose-s_aist
```

## Step 4: Inference

Run inference on new images:

```bash
python tools/inference.py \
    configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py \
    checkpoints/rtmpose-s_simcc-coco.pth \
    --input-path path/to/images \
    --output-path vis_results
```

## Dataset Format

The converted dataset follows COCO format:
- **Images**: Frame images with naming `{video_name}_frame_{index:06d}.jpg`
- **Annotations**: COCO JSON format with:
  - 17 keypoints (COCO format)
  - Bounding boxes
  - Visibility flags
  - Detection scores

## Key Features

- ✅ Frames aligned with annotations (1:1 mapping)
- ✅ COCO format compatible with RTM Pose
- ✅ Automatic bounding box generation
- ✅ Confidence filtering
- ✅ Frame-to-annotation mapping preserved

## Troubleshooting

**Frame count mismatch:**
- Ensure frames are extracted with `--alignWithAnnotations` flag
- Check that frame extraction completed successfully

**Missing annotations:**
- Verify keypoints2d files exist for all videos
- Check that annotation files are valid pickle files

**RTM Pose errors:**
- Ensure MMPose is properly installed
- Check that config paths match your dataset structure
- Verify COCO format JSON is valid


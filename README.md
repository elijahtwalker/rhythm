# Rhythm!

Simple tools for downloading, organizing, and working with the AIST++ dataset.

## Quick Start

### 1. Check what you have
```bash
python3 checkDataStatus.py
```

### 2. Download videos
```bash
# Download all splits (train, val, test)
python3 downloadSplitVideos.py --split all --numProcesses 4

# Or download a specific split
python3 downloadSplitVideos.py --split pose_train --numProcesses 4
```

### 3. Load data
```bash
# Get statistics
python3 loadAistData.py --annoDir . --stats

# Load training data
python3 loadAistData.py --annoDir . --split train
```

## Scripts

- **`downloadSplitVideos.py`** - Download videos organized by splits
- **`checkDataStatus.py`** - Check download status and what's missing
- **`loadAistData.py`** - Load data using AIST++ API structure
- **`visualizeKeypoints.py`** - Visualize 2D keypoints

## Directory Structure

After downloading:
```
rhythm/
├── splits/
│   ├── pose_train.txt
│   ├── pose_val.txt
│   └── pose_test.txt
├── videos/
│   ├── pose_train/    # Training videos
│   ├── pose_val/      # Validation videos
│   └── pose_test/     # Test videos
└── keypoints2d/       # 2D keypoints (already present)
```

## Notes

- Videos are automatically organized into split directories
- Already downloaded videos are skipped
- Failed downloads are saved to `*_failed.txt` files
- You must agree to AIST Terms of Use before downloading
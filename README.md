# Rhythm!

End-to-end workflow for preparing the AIST++ dataset, generating aligned frames, converting to COCO/RTM Pose, and visualizing overlays.

---

## 1. Environment Setup (Required)

We hit segmentation faults with the system Python, so everything runs inside a dedicated virtual environment.

```bash
cd /YOUR/PATH/rhythm
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate
pip install numpy==1.26.4 pillow==10.3.0 ffmpeg-python==0.2.0 opencv-python==4.10.0.84
```

Activate the env (`source .venv/bin/activate`) before running any script. Run `deactivate` when finished.

---

## 2. Dataset Preparation Workflow

| Step | Required? | Command / Notes |
|------|-----------|----------------|
|Check current assets|Optional but helpful|`python3 checkDataStatus.py`|
|Download videos|**Required** if you need frames/RTM Pose|`python3 downloadSplitVideos.py --split pose_train --numProcesses 4`<br/>If CDN URLs 404, download manually from [AIST Dance DB](https://aistdancedb.ongaaccel.jp/). Skip only if videos already exist locally.|
|Inspect splits / statistics|Optional|`python3 loadAistData.py --annoDir . --stats`|
|Extract frames aligned with annotations|**Required** for RTM Pose/workflows needing images|`python3 extractVideoFrames.py --split pose_train --alignWithAnnotations`<br/>Repeat for `pose_val` / `pose_test` if needed. Generates JPEG frames plus `frame_mapping.json` metadata using FFmpeg (exact 60 FPS or annotation timestamps).|
|Convert to RTM Pose COCO format|**Required** for training|`python3 convertToRTMPose.py --split pose_train --outputDir rtmpose_dataset`<br/>Run for each split you plan to use.|
|Overlay QA (stick figures)|Optional|`python3 visualizeAlignedAnnotations.py --videoName <video_id> --framesDir frames --keypoints2dDir keypoints2d --outputDir overlays/<video_id>_colored --limit 20`<br/>Uses detection scores + visibility heuristics to filter ghost tracks.|

> `ignore_list.txt` tracks problematic videos (missing frames/annotations). Skip those when extracting or visualizing.

---

## 3. Directory Structure

```
rhythm/
├── keypoints2d/
├── splits/
│   ├── pose_train.txt
│   ├── pose_val.txt
│   └── pose_test.txt
├── videos/
│   ├── pose_train/
│   ├── pose_val/
│   └── pose_test/
├── frames/<video_id>/          # generated JPEGs + frame_mapping.json
├── overlays/<video_id>_colored # optional QA renders
└── rtmpose_dataset/            # COCO-format output (images + annotations)
```

`rtmpose_dataset` layout:

```
rtmpose_dataset/
├── train/
│   ├── images/
│   └── annotations/train.json
├── val/
│   ├── images/
│   └── annotations/val.json
└── test/
    ├── images/
    └── annotations/test.json
```

---

## 4. RTM Pose Integration

1. **Prepare data** (Section 2). Frames + `rtmpose_dataset` must exist.
2. **Install MMPose/RTM Pose** (outside this repo):
   ```bash
   git clone https://github.com/open-mmlab/mmpose.git
   cd mmpose
   pip install -v -e .
   ```
3. **Configure dataset paths:**
   ```python
   dataset_type = 'CocoDataset'
   data_root = '/path/to/rhythm/rtmpose_dataset'
   ann_file = 'train/annotations/train.json'
   data_prefix = dict(img='train/images/')
   ```
4. **Train:**
   ```bash
   python tools/train.py \
       configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py \
       --work-dir work_dirs/rtmpose-s_aist
   ```
5. **Inference:**
   ```bash
   python tools/inference.py \
       configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py \
       checkpoints/rtmpose-s_simcc-coco.pth \
       --input-path path/to/images \
       --output-path vis_results
   ```

---

## 5. Script Reference

| Script | Purpose |
|--------|---------|
|`checkDataStatus.py`|Summarize downloaded videos and keypoints per split.|
|`downloadSplitVideos.py`|Download videos per split; handles retries/404 messaging.|
|`loadAistData.py`|Loader matching the official AIST++ API; inspect stats or read pickles.|
|`extractVideoFrames.py`|FFmpeg-based extractor (60 FPS or annotation timestamps) with `frame_mapping.json`.|
|`convertToRTMPose.py`|Build COCO-format dataset (images + annotations + bounding boxes).|
|`visualizeKeypoints.py`|Quick matplotlib viewer for `.pkl` files.|
|`visualizeAlignedAnnotations.py`|Overlay frames with annotations, filtering low-score tracks.|

---

## 6. Optional QA Tips

- Regenerate overlays after re-extracting frames to confirm alignment.
- `--limit 0` on `visualizeAlignedAnnotations.py` renders the entire video.
- Adjust `--minConfidence` if you want stricter limb drawing (default 0.1).

---

## 7. Troubleshooting

- **Videos missing / 404** – Use the official [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/) and copy files into `videos/<split>/`.
- **Frame count mismatches** – Delete `frames/<video>` and re-run `extractVideoFrames.py --alignWithAnnotations`.
- **Ghost stick figures** – Ensure you’re using the latest overlay script; it keeps only the highest detection-score track per frame.
- **NumPy segmentation faults** – Always run inside `.venv` (see Section 1).
- **COCO conversion errors** – Make sure frames exist for every video listed in the split and that `keypoints2d/*.pkl` files are present.

---

Following this guide from top to bottom reproduces the entire pipeline:

1. Set up the virtual environment (required).
2. Download videos if you need images (optional).
3. Extract frames with alignment (required for RTM Pose / overlays).
4. Convert to RTM Pose (required for training).
5. Train/infer with RTM Pose, or optionally visualize overlays for QA.

Everything else—extra stats, overlays, selective downloads—remains optional and is clearly labeled above. Enjoy!***
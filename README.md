# Rhythm: Pose Detection for Multi-Genre Dance Choreography

End-to-end workflow for preparing the AIST++ dataset, generating aligned frames, running MediaPipe Pose, and visualizing overlays.

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
|Download videos|**Required** if you need frames|`python3 downloadSplitVideos.py --split pose_train --numProcesses 4`<br/>If CDN URLs 404, download manually from [AIST Dance DB](https://aistdancedb.ongaaccel.jp/). Skip only if videos already exist locally.|
|Inspect splits / statistics|Optional|`python3 loadAistData.py --annoDir . --stats`|
|Extract frames aligned with annotations|**Required** for MediaPipe/workflows needing images|`python3 extractVideoFrames.py --split pose_train --alignWithAnnotations`<br/>Repeat for `pose_val` / `pose_test` if needed. Generates JPEG frames plus `frame_mapping.json` metadata using FFmpeg (exact 60 FPS or annotation timestamps).|
|Overlay QA (stick figures)|Optional|`python3 visualizeAlignedAnnotations.py --videoName <video_id> --framesDir frames --keypoints2dDir keypoints2d --outputDir overlays/<video_id>_colored --limit 20`<br/>Uses detection scores + visibility heuristics to filter ghost tracks.|
|Evaluate pose vs. ground truth|Optional but recommended|`python3 calculateOKS.py` for OKS summaries and per-keypoint detection stats (requires `keypoints2d/*.pkl` + `keypoints_mediapipe/*_keypoints.json`).<br/>`python3 calculatePDJ.py` for PDJ summaries using 0.05 × bbox diagonal as the threshold.|

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
├── frames/<video_id>/
└── overlays/<video_id>_colored
```

---

## 4. Colab Notebook: `model_train_test_script.ipynb`

Use this notebook as a reference for cloud/GPU preprocessing or training:

- Mount Google Drive and unzip `frames.zip` to `/content/drive/MyDrive/rhythm_frames`.
- Install runtime deps inside the notebook session: `mediapipe`, `opencv-python`, `tqdm`.
- Set output folders (for example `/content/pose_keypoints` or `/content/pose_keypoints_single_video`).
- Run the MediaPipe Pose loop to generate per-video JSON keypoint dumps (`<video>/<video>_keypoints.json`).
- Optional cells show how to draw pose overlays on sample frames for quick QA.

---

## 5. Script Reference

| Script | Purpose |
|--------|---------|
|`checkDataStatus.py`|Summarize downloaded videos and keypoints per split.|
|`downloadSplitVideos.py`|Download videos per split; handles retries/404 messaging.|
|`loadAistData.py`|Loader matching the official AIST++ API; inspect stats or read pickles.|
|`extractVideoFrames.py`|FFmpeg-based extractor (60 FPS or annotation timestamps) with `frame_mapping.json`.|
|`visualizeKeypoints.py`|Quick matplotlib viewer for `.pkl` files.|
|`visualizeAlignedAnnotations.py`|Overlay frames with annotations, filtering low-score tracks.|
|`calculateOKS.py`|Compute OKS per frame/keypoint and global averages vs. MediaPipe outputs.|
|`calculatePDJ.py`|Compute PDJ (0.05 × bbox diagonal) detection rates vs. MediaPipe outputs.|

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
3. Extract frames with alignment (required for MediaPipe / overlays).
4. Use the Colab notebook for MediaPipe-based preprocessing or quick runs if you prefer GPU-in-the-cloud.
5. Optionally visualize overlays for QA.
6. Optionally evaluate MediaPipe vs. ground truth with `calculateOKS.py` / `calculatePDJ.py`.

Everything else—extra stats, overlays, selective downloads—remains optional and is clearly labeled above. Enjoy!
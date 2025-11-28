#!/usr/bin/env python3
"""
Extract frames from videos at exactly 60 FPS.
Converts videos to image sequences, resampling to 60 FPS regardless of original FPS.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import ffmpeg
import numpy as np


def parseFpsValue(fpsString):
    """Parse fps string from ffmpeg metadata (e.g., '30000/1001')."""
    if not fpsString or fpsString in ('0/0', '0'):
        return 0.0
    if '/' in fpsString:
        num, den = fpsString.split('/')
        den = float(den) if float(den) != 0 else 1.0
        return float(num) / den
    return float(fpsString)


def ffmpegVideoRead(videoPath, targetFPS=None):
    """
    Read video frames using ffmpeg, optionally resampling to targetFPS.
    
    Returns:
        frames (np.ndarray): shape (num_frames, H, W, 3) in RGB
        originalFPS (float): original video FPS from metadata
    """
    videoPath = Path(videoPath)
    if not videoPath.exists():
        raise FileNotFoundError(f"{videoPath} does not exist")
    
    probe = ffmpeg.probe(str(videoPath))
    videoStream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    
    width = int(videoStream['width'])
    height = int(videoStream['height'])
    originalFPS = parseFpsValue(videoStream.get('avg_frame_rate')) or parseFpsValue(videoStream.get('r_frame_rate'))
    
    stream = ffmpeg.input(str(videoPath))
    if targetFPS:
        stream = stream.filter('fps', fps=targetFPS, round='down')
    
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
    
    frameArray = np.frombuffer(out, np.uint8)
    frameSize = height * width * 3
    if frameArray.size % frameSize != 0:
        raise ValueError("FFmpeg output size is not divisible by frame dimensions; video may be corrupted.")
    
    frameCount = frameArray.size // frameSize
    frames = frameArray.reshape((frameCount, height, width, 3))
    return frames, float(originalFPS)


def extractFramesAt60FPS(videoPath, outputDir, targetFPS=60, annotationPath=None):
    """
    Extract frames from video at exactly targetFPS (default 60 FPS).
    If annotationPath is provided, extracts frames matching annotation timestamps.
    
    Args:
        videoPath: Path to input video file
        outputDir: Directory to save extracted frames
        targetFPS: Target FPS (default: 60)
        annotationPath: Optional path to keypoints2d pickle file for alignment
    
    Returns:
        Dictionary with extraction info (frameCount, timestamps, etc.)
    """
    videoPath = Path(videoPath)
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations if provided
    timestamps = None
    annotationFrameCount = None
    if annotationPath and Path(annotationPath).exists():
        with open(annotationPath, 'rb') as f:
            annotationData = pickle.load(f)
            rawTimestamps = annotationData.get('timestamps', None)
            if rawTimestamps is not None:
                timestamps = np.asarray(rawTimestamps).astype(float).tolist()
                annotationFrameCount = len(timestamps)
                print(f"  Found {annotationFrameCount} annotation frames")
    
    frames, originalFPS = ffmpegVideoRead(videoPath, targetFPS)
    targetFPSUsed = targetFPS if targetFPS else originalFPS
    duration = len(frames) / targetFPSUsed if targetFPSUsed else 0
    
    print(f"Video: {videoPath.name}")
    print(f"  Original FPS: {originalFPS:.2f}")
    print(f"  Frames after resample: {len(frames)}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting at {targetFPSUsed:.2f} FPS...")
    
    # Determine timestamp targets
    if timestamps is not None:
        targetTimes = [ts / 1000.0 for ts in timestamps]
        print(f"  Aligning with {len(targetTimes)} annotation timestamps")
    else:
        targetTimes = [i / targetFPSUsed for i in range(len(frames))]
    
    frameMapping = {}
    extractedTimestamps = []
    
    savedCount = 0
    maxFrames = min(len(frames), len(targetTimes))
    if len(frames) != len(targetTimes):
        print(f"  Note: Frame count ({len(frames)}) and timestamp count ({len(targetTimes)}) differ; using {maxFrames} aligned frames.")
    
    for frameIndex in range(maxFrames):
        frame = frames[frameIndex]
        targetTime = targetTimes[frameIndex]
        framePath = outputDir / f"frame_{frameIndex:06d}.jpg"
        
        # Convert RGB to BGR for OpenCV writing
        frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(framePath), frameBGR)
        
        frameMapping[frameIndex] = {
            'filename': framePath.name,
            'timestamp_ms': int(targetTime * 1000),
            'timestamp_sec': targetTime,
            'annotation_index': frameIndex if timestamps is not None else None
        }
        extractedTimestamps.append(int(targetTime * 1000))
        savedCount += 1
    
    # Save frame mapping metadata
    metadataPath = outputDir / 'frame_mapping.json'
    with open(metadataPath, 'w') as f:
        json.dump({
            'video_name': videoPath.stem,
            'original_fps': float(originalFPS),
            'target_fps': targetFPSUsed,
            'total_frames': savedCount,
            'frame_mapping': frameMapping,
            'timestamps_ms': extractedTimestamps,
            'aligned_with_annotations': timestamps is not None
        }, f, indent=2)
    
    print(f"  Extracted {savedCount} frames at {targetFPS} FPS")
    print(f"  Frames saved to: {outputDir}")
    print(f"  Frame mapping saved to: {metadataPath}")
    
    if timestamps is not None and savedCount != annotationFrameCount:
        print(f"  ⚠ Warning: Extracted {savedCount} frames but annotations have {annotationFrameCount} frames")
    
    return {
        'frameCount': savedCount,
        'timestamps': extractedTimestamps,
        'mapping': frameMapping
    }


def resolveVideoPath(videoName, videoDir, splitName=None):
    """
    Resolve the actual path to a video, checking split subdirectories.
    """
    videoDir = Path(videoDir)
    
    candidateDirs = []
    if splitName:
        candidateDirs.append(videoDir / splitName)
    candidateDirs.append(videoDir)
    
    for baseDir in candidateDirs:
        candidatePath = baseDir / f"{videoName}.mp4"
        if candidatePath.exists():
            return candidatePath
    
    return None


def extractFramesForVideo(videoName, videoDir, outputBaseDir, targetFPS=60, keypoints2dDir=None, splitName=None):
    """
    Extract frames for a single video, aligned with annotations if available.
    
    Args:
        videoName: Video name (without extension)
        videoDir: Directory containing videos
        outputBaseDir: Base directory for output frames
        targetFPS: Target FPS (default: 60)
        keypoints2dDir: Optional directory containing keypoints2d annotations
    """
    videoPath = resolveVideoPath(videoName, videoDir, splitName)
    
    if videoPath is None:
        searchLocations = [Path(videoDir)]
        if splitName:
            searchLocations.insert(0, Path(videoDir) / splitName)
        print(f"✗ Video not found: looked in {[str(p) for p in searchLocations]}")
        return False
    
    outputDir = Path(outputBaseDir) / videoName
    
    # Try to find corresponding annotation file
    annotationPath = None
    if keypoints2dDir:
        annotationPath = Path(keypoints2dDir) / f"{videoName}.pkl"
        if not annotationPath.exists():
            annotationPath = None
    
    try:
        extractFramesAt60FPS(videoPath, outputDir, targetFPS, annotationPath)
        return True
    except Exception as e:
        print(f"✗ Error extracting {videoName}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from videos at exactly 60 FPS')
    parser.add_argument(
        '--videoPath',
        type=str,
        default=None,
        dest='videoPath',
        help='Path to single video file to process')
    parser.add_argument(
        '--videoName',
        type=str,
        default=None,
        dest='videoName',
        help='Video name (without extension) to process')
    parser.add_argument(
        '--videoDir',
        type=str,
        default='videos',
        dest='videoDir',
        help='Directory containing videos (default: videos)')
    parser.add_argument(
        '--outputDir',
        type=str,
        default='frames',
        dest='outputDir',
        help='Output directory for frames (default: frames)')
    parser.add_argument(
        '--targetFPS',
        type=int,
        default=60,
        dest='targetFPS',
        help='Target FPS for frame extraction (default: 60)')
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['pose_train', 'pose_val', 'pose_test'],
        help='Process all videos in a split')
    parser.add_argument(
        '--splitsDir',
        type=str,
        default='splits',
        dest='splitsDir',
        help='Directory containing split files (default: splits)')
    parser.add_argument(
        '--keypoints2dDir',
        type=str,
        default='keypoints2d',
        dest='keypoints2dDir',
        help='Directory containing keypoints2d annotations for alignment (default: keypoints2d)')
    parser.add_argument(
        '--alignWithAnnotations',
        action='store_true',
        help='Align extracted frames with annotation timestamps (requires --keypoints2dDir)')
    
    args = parser.parse_args()
    
    keypoints2dDir = args.keypoints2dDir if args.alignWithAnnotations else None
    
    if args.videoPath:
        # Process single video file
        outputDir = Path(args.outputDir) / Path(args.videoPath).stem
        annotationPath = None
        if args.alignWithAnnotations:
            videoName = Path(args.videoPath).stem
            annotationPath = Path(args.keypoints2dDir) / f"{videoName}.pkl"
        extractFramesAt60FPS(args.videoPath, outputDir, args.targetFPS, annotationPath)
    elif args.videoName:
        # Process single video by name
        extractFramesForVideo(args.videoName, args.videoDir, args.outputDir, args.targetFPS, keypoints2dDir)
    elif args.split:
        # Process all videos in a split
        splitsDir = Path(args.splitsDir)
        splitFile = splitsDir / f"{args.split}.txt"
        
        if not splitFile.exists():
            print(f"Error: Split file not found: {splitFile}")
            return
        
        with open(splitFile, 'r') as f:
            videoNames = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(videoNames)} videos from {args.split} split...")
        print("=" * 60)
        
        successful = 0
        for i, videoName in enumerate(videoNames):
            print(f"\n[{i+1}/{len(videoNames)}] Processing {videoName}...")
            if extractFramesForVideo(
                videoName,
                args.videoDir,
                args.outputDir,
                args.targetFPS,
                keypoints2dDir,
                splitName=args.split
            ):
                successful += 1
        
        print("\n" + "=" * 60)
        print(f"Extraction complete: {successful}/{len(videoNames)} videos processed")
    else:
        parser.print_help()
        print("\nError: Must specify --videoPath, --videoName, or --split")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Extract frames from videos at exactly 60 FPS.
Converts videos to image sequences, resampling to 60 FPS regardless of original FPS.
"""

import cv2
import argparse
from pathlib import Path
import numpy as np
import pickle
import json


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
            timestamps = annotationData.get('timestamps', None)
            if timestamps is not None:
                annotationFrameCount = len(timestamps)
                print(f"  Found {annotationFrameCount} annotation frames")
    
    # Open video
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")
    
    # Get video properties
    originalFPS = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = totalFrames / originalFPS if originalFPS > 0 else 0
    
    print(f"Video: {videoPath.name}")
    print(f"  Original FPS: {originalFPS:.2f}")
    print(f"  Total frames: {totalFrames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting at {targetFPS} FPS...")
    
    # If we have annotation timestamps, use them for frame extraction
    if timestamps is not None:
        # Timestamps are in milliseconds, convert to seconds
        targetTimes = [ts / 1000.0 for ts in timestamps]
        print(f"  Aligning with {len(targetTimes)} annotation timestamps")
    else:
        # Calculate frame times at target FPS
        expectedFrames = int(duration * targetFPS)
        targetTimes = [i / targetFPS for i in range(expectedFrames)]
    
    frameCount = 0
    savedCount = 0
    frameMapping = {}  # Maps annotation frame index to extracted frame filename
    extractedTimestamps = []
    
    # Read through video and extract frames at target times
    for targetTime in targetTimes:
        # Seek to target time
        cap.set(cv2.CAP_PROP_POS_MSEC, targetTime * 1000)
        ret, frame = cap.read()
        
        if not ret:
            # If we can't read at exact time, try to get closest frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(targetTime * originalFPS))
            ret, frame = cap.read()
        
        if ret:
            # Save frame with annotation-aligned index
            frameIndex = savedCount
            framePath = outputDir / f"frame_{frameIndex:06d}.jpg"
            cv2.imwrite(str(framePath), frame)
            
            # Store mapping
            frameMapping[frameIndex] = {
                'filename': framePath.name,
                'timestamp_ms': int(targetTime * 1000),
                'timestamp_sec': targetTime,
                'annotation_index': frameIndex if timestamps else None
            }
            extractedTimestamps.append(int(targetTime * 1000))
            savedCount += 1
        else:
            print(f"  Warning: Could not extract frame at {targetTime:.3f}s")
    
    cap.release()
    
    # Save frame mapping metadata
    metadataPath = outputDir / 'frame_mapping.json'
    with open(metadataPath, 'w') as f:
        json.dump({
            'video_name': videoPath.stem,
            'original_fps': float(originalFPS),
            'target_fps': targetFPS,
            'total_frames': savedCount,
            'frame_mapping': frameMapping,
            'timestamps_ms': extractedTimestamps,
            'aligned_with_annotations': timestamps is not None
        }, f, indent=2)
    
    print(f"  Extracted {savedCount} frames at {targetFPS} FPS")
    print(f"  Frames saved to: {outputDir}")
    print(f"  Frame mapping saved to: {metadataPath}")
    
    if timestamps and savedCount != annotationFrameCount:
        print(f"  ⚠ Warning: Extracted {savedCount} frames but annotations have {annotationFrameCount} frames")
    
    return {
        'frameCount': savedCount,
        'timestamps': extractedTimestamps,
        'mapping': frameMapping
    }


def extractFramesForVideo(videoName, videoDir, outputBaseDir, targetFPS=60, keypoints2dDir=None):
    """
    Extract frames for a single video, aligned with annotations if available.
    
    Args:
        videoName: Video name (without extension)
        videoDir: Directory containing videos
        outputBaseDir: Base directory for output frames
        targetFPS: Target FPS (default: 60)
        keypoints2dDir: Optional directory containing keypoints2d annotations
    """
    videoPath = Path(videoDir) / f"{videoName}.mp4"
    
    if not videoPath.exists():
        print(f"✗ Video not found: {videoPath}")
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
            if extractFramesForVideo(videoName, args.videoDir, args.outputDir, args.targetFPS, keypoints2dDir):
                successful += 1
        
        print("\n" + "=" * 60)
        print(f"Extraction complete: {successful}/{len(videoNames)} videos processed")
    else:
        parser.print_help()
        print("\nError: Must specify --videoPath, --videoName, or --split")


if __name__ == '__main__':
    main()


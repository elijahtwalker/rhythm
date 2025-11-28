#!/usr/bin/env python3
"""
Overlay AIST++ keypoint annotations onto extracted frames to verify alignment.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


COCO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Right arm
    (0, 5), (5, 6), (6, 7), (7, 8),        # Left arm
    (0, 9), (9, 10), (10, 11),             # Right leg
    (0, 12), (12, 13), (13, 14),           # Left leg
    (0, 15), (0, 16), (15, 17), (16, 18)   # Head/eyes (AIST uses 17 points; last pair ignored if missing)
]


def loadKeypoints(keypointsPath):
    with open(keypointsPath, 'rb') as f:
        data = pickle.load(f)
    return data.get('keypoints2d')


def loadFrameMapping(framesDir, videoName):
    mappingPath = Path(framesDir) / videoName / 'frame_mapping.json'
    if mappingPath.exists():
        with open(mappingPath, 'r') as f:
            mappingData = json.load(f)
        # Mapping keys are string indices
        entries = []
        for idxStr, meta in mappingData.get('frame_mapping', {}).items():
            entries.append((int(idxStr), meta['filename']))
        if entries:
            return sorted(entries, key=lambda x: x[0])
    # Fallback: sequential filenames
    frameFiles = sorted((Path(framesDir) / videoName).glob('frame_*.jpg'))
    return [(i, frame.name) for i, frame in enumerate(frameFiles)]


def drawPerson(image, keypoints, minConfidence=0.3):
    vis = image.copy()
    points = []
    for idx, (x, y, conf) in enumerate(keypoints):
        points.append((int(x), int(y), conf))
        if conf >= minConfidence:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    for start, end in COCO_SKELETON:
        if start >= len(points) or end >= len(points):
            continue
        x1, y1, c1 = points[start]
        x2, y2, c2 = points[end]
        if c1 >= minConfidence and c2 >= minConfidence:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
    return vis


def overlayAnnotations(videoName, framesDir, keypointsDir, outputDir,
                       maxFrames=None, minConfidence=0.3):
    keypointsPath = Path(keypointsDir) / f"{videoName}.pkl"
    if not keypointsPath.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypointsPath}")
    
    keypointsData = loadKeypoints(keypointsPath)
    if keypointsData is None:
        raise ValueError(f"No keypoints2d data in {keypointsPath}")
    
    mapping = loadFrameMapping(framesDir, videoName)
    if not mapping:
        raise FileNotFoundError(f"No frames found for {videoName} in {framesDir}")
    
    videoFramesDir = Path(framesDir) / videoName
    outputVideoDir = Path(outputDir) / videoName
    outputVideoDir.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    for frameIdx, frameFilename in mapping:
        if maxFrames and processed >= maxFrames:
            break
        
        if frameIdx >= len(keypointsData):
            break
        
        framePath = videoFramesDir / frameFilename
        if not framePath.exists():
            continue
        
        image = cv2.imread(str(framePath))
        if image is None:
            continue
        
        peopleKeypoints = keypointsData[frameIdx]
        overlay = image.copy()
        for personKeypoints in peopleKeypoints:
            overlay = drawPerson(overlay, personKeypoints, minConfidence=minConfidence)
        
        savePath = outputVideoDir / frameFilename
        cv2.imwrite(str(savePath), overlay)
        processed += 1
    
    return processed, len(mapping)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay keypoint annotations onto extracted frames")
    parser.add_argument(
        '--videoName',
        type=str,
        required=True,
        help='Video name (without extension) to visualize')
    parser.add_argument(
        '--framesDir',
        type=str,
        default='frames',
        help='Directory containing extracted frames (default: frames)')
    parser.add_argument(
        '--keypoints2dDir',
        type=str,
        default='keypoints2d',
        help='Directory containing keypoints2d pickle files (default: keypoints2d)')
    parser.add_argument(
        '--outputDir',
        type=str,
        default='frames_with_annotations',
        help='Directory to save overlay images (default: frames_with_annotations)')
    parser.add_argument(
        '--maxFrames',
        type=int,
        default=None,
        help='Optional limit on number of frames to process')
    parser.add_argument(
        '--minConfidence',
        type=float,
        default=0.3,
        help='Confidence threshold for drawing keypoints/skeleton')
    
    args = parser.parse_args()
    
    processed, total = overlayAnnotations(
        videoName=args.videoName,
        framesDir=args.framesDir,
        keypointsDir=args.keypoints2dDir,
        outputDir=args.outputDir,
        maxFrames=args.maxFrames,
        minConfidence=args.minConfidence
    )
    
    print(f"Overlay complete: {processed}/{total} frames written to {args.outputDir}/{args.videoName}")


if __name__ == '__main__':
    main()


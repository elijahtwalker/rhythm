#!/usr/bin/env python3
"""
Convert AIST++ dataset to RTM Pose format.
Creates dataset structure compatible with MMPose RTM Pose training and inference.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


# COCO keypoint format (17 keypoints)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton connections (1-indexed for COCO format)
# Format: [start_keypoint_idx, end_keypoint_idx] where indices are 1-based
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [12, 6], [13, 7], [12, 13],
    [1, 2], [1, 3], [2, 4], [3, 5]
]


def convertKeypointsToCOCOFormat(keypoints2d, detScores=None, minConfidence=0.3):
    """
    Convert AIST++ keypoints to COCO format.
    
    Args:
        keypoints2d: Array of shape (num_frames, num_people, num_keypoints, 3)
        detScores: Optional detection scores array
        minConfidence: Minimum confidence threshold
    
    Returns:
        List of COCO-format annotations per frame
    """
    cocoAnnotations = []
    
    for frameIdx, frameKeypoints in enumerate(keypoints2d):
        # frameKeypoints shape: (num_people, num_keypoints, 3)
        for personIdx, personKeypoints in enumerate(frameKeypoints):
            # personKeypoints shape: (num_keypoints, 3) where 3 is (x, y, confidence)
            
            # Filter by confidence
            confidences = personKeypoints[:, 2]
            validKeypoints = confidences > minConfidence
            
            if np.sum(validKeypoints) < 5:  # Need at least 5 valid keypoints
                continue
            
            # Convert to COCO format: [x1, y1, v1, x2, y2, v2, ...]
            # v = 0: not labeled, 1: labeled but not visible, 2: labeled and visible
            cocoKeypoints = []
            for kp in personKeypoints:
                x, y, conf = kp
                if conf > minConfidence:
                    visibility = 2 if conf > 0.5 else 1
                else:
                    visibility = 0
                cocoKeypoints.extend([float(x), float(y), visibility])
            
            # Calculate bounding box from keypoints
            validX = personKeypoints[validKeypoints, 0]
            validY = personKeypoints[validKeypoints, 1]
            
            if len(validX) > 0 and len(validY) > 0:
                xMin, xMax = float(np.min(validX)), float(np.max(validX))
                yMin, yMax = float(np.min(validY)), float(np.max(validY))
                
                # Add padding
                padding = 20
                bbox = [
                    max(0, xMin - padding),
                    max(0, yMin - padding),
                    xMax - xMin + 2 * padding,
                    yMax - yMin + 2 * padding
                ]
                
                # Get detection score if available
                area = bbox[2] * bbox[3]
                score = float(detScores[frameIdx][personIdx]) if detScores is not None else float(np.mean(confidences))
                
                annotation = {
                    'id': len(cocoAnnotations),
                    'image_id': frameIdx,
                    'category_id': 1,  # Person category
                    'keypoints': cocoKeypoints,
                    'num_keypoints': int(np.sum(validKeypoints)),
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0,
                    'score': score
                }
                cocoAnnotations.append(annotation)
    
    return cocoAnnotations


def createRTMPoseDataset(videoName: str,
                        framesDir: str,
                        keypoints2dDir: str,
                        outputDir: str,
                        split: str = 'train',
                        minConfidence: float = 0.3):
    """
    Create RTM Pose dataset structure for a single video.
    
    Args:
        videoName: Video name (without extension)
        framesDir: Directory containing extracted frames
        keypoints2dDir: Directory containing keypoints2d annotations
        outputDir: Output directory for RTM Pose dataset
        split: Dataset split ('train', 'val', 'test')
        minConfidence: Minimum confidence threshold for keypoints
    """
    outputDir = Path(outputDir)
    splitDir = outputDir / split
    imagesDir = splitDir / 'images'
    annotationsDir = splitDir / 'annotations'
    
    imagesDir.mkdir(parents=True, exist_ok=True)
    annotationsDir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotationPath = Path(keypoints2dDir) / f"{videoName}.pkl"
    if not annotationPath.exists():
        print(f"  ⚠ Annotations not found: {annotationPath}")
        return None
    
    with open(annotationPath, 'rb') as f:
        annotationData = pickle.load(f)
    
    keypoints2d = annotationData.get('keypoints2d', None)
    detScores = annotationData.get('det_scores', None)
    timestamps = annotationData.get('timestamps', None)
    
    if keypoints2d is None:
        print(f"  ⚠ No keypoints2d found for {videoName}")
        return None
    
    # Load frame mapping if available
    frameMappingPath = Path(framesDir) / videoName / 'frame_mapping.json'
    frameMapping = None
    if frameMappingPath.exists():
        with open(frameMappingPath, 'r') as f:
            frameMapping = json.load(f)
    
    # Get frame files
    framesPath = Path(framesDir) / videoName
    if not framesPath.exists():
        print(f"  ⚠ Frames directory not found: {framesPath}")
        return None
    
    frameFiles = sorted(framesPath.glob('frame_*.jpg'))
    if len(frameFiles) != len(keypoints2d):
        print(f"  ⚠ Frame count mismatch: {len(frameFiles)} frames vs {len(keypoints2d)} annotations")
    
    # Convert to COCO format
    cocoAnnotations = convertKeypointsToCOCOFormat(keypoints2d, detScores, minConfidence)
    
    # Create COCO format dataset
    images = []
    annotations = []
    
    for frameIdx, frameFile in enumerate(frameFiles):
        # Copy/link frame to output directory
        imageId = len(images)
        imageFilename = f"{videoName}_frame_{frameIdx:06d}.jpg"
        outputImagePath = imagesDir / imageFilename
        
        # Create symlink or copy
        try:
            if outputImagePath.exists():
                outputImagePath.unlink()
            outputImagePath.symlink_to(frameFile.resolve())
        except:
            # If symlink fails, copy the file
            import shutil
            shutil.copy2(frameFile, outputImagePath)
        
            # Get image dimensions
            try:
                import cv2
                img = cv2.imread(str(frameFile))
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    height, width = 1080, 1920  # Default dimensions
            except:
                # If cv2 not available, use default dimensions
                height, width = 1080, 1920
        
        images.append({
            'id': imageId,
            'file_name': imageFilename,
            'width': width,
            'height': height
        })
        
        # Update annotation image_ids
        frameAnnotations = [ann for ann in cocoAnnotations if ann['image_id'] == frameIdx]
        for ann in frameAnnotations:
            ann['image_id'] = imageId
            ann['id'] = len(annotations)
            annotations.append(ann)
    
    return {
        'images': images,
        'annotations': annotations,
        'video_name': videoName
    }


def createRTMPoseDatasetFromSplit(split: str,
                                  splitsDir: str,
                                  framesDir: str,
                                  keypoints2dDir: str,
                                  outputDir: str,
                                  minConfidence: float = 0.3):
    """
    Create RTM Pose dataset for all videos in a split.
    
    Args:
        split: Split name ('pose_train', 'pose_val', 'pose_test')
        splitsDir: Directory containing split files
        framesDir: Directory containing extracted frames
        keypoints2dDir: Directory containing keypoints2d annotations
        outputDir: Output directory for RTM Pose dataset
        minConfidence: Minimum confidence threshold
    """
    splitsDir = Path(splitsDir)
    splitFile = splitsDir / f"{split}.txt"
    
    if not splitFile.exists():
        raise FileNotFoundError(f"Split file not found: {splitFile}")
    
    with open(splitFile, 'r') as f:
        videoNames = [line.strip() for line in f if line.strip()]
    
    print(f"Converting {len(videoNames)} videos from {split} to RTM Pose format...")
    print("=" * 70)
    
    allImages = []
    allAnnotations = []
    imageIdOffset = 0
    annotationIdOffset = 0
    
    successful = 0
    for videoName in tqdm(videoNames, desc="Processing videos"):
        result = createRTMPoseDataset(
            videoName,
            framesDir,
            keypoints2dDir,
            outputDir,
            split=split.replace('pose_', ''),
            minConfidence=minConfidence
        )
        
        if result:
            # Update IDs to be globally unique
            for img in result['images']:
                img['id'] += imageIdOffset
                allImages.append(img)
            
            for ann in result['annotations']:
                ann['id'] += annotationIdOffset
                ann['image_id'] += imageIdOffset
                allAnnotations.append(ann)
            
            imageIdOffset += len(result['images'])
            annotationIdOffset += len(result['annotations'])
            successful += 1
    
    # Create COCO format JSON
    cocoDataset = {
        'info': {
            'description': f'AIST++ Dataset - {split}',
            'version': '1.0',
            'year': 2024
        },
        'licenses': [],
        'categories': [{
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
            'keypoints': COCO_KEYPOINT_NAMES,
            'skeleton': COCO_SKELETON
        }],
        'images': allImages,
        'annotations': allAnnotations
    }
    
    # Save annotation file
    outputDir = Path(outputDir)
    splitName = split.replace('pose_', '')
    annotationsPath = outputDir / splitName / 'annotations' / f'{splitName}.json'
    annotationsPath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(annotationsPath, 'w') as f:
        json.dump(cocoDataset, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Conversion complete!")
    print(f"  Videos processed: {successful}/{len(videoNames)}")
    print(f"  Total images: {len(allImages)}")
    print(f"  Total annotations: {len(allAnnotations)}")
    print(f"  Annotation file: {annotationsPath}")
    print(f"  Images directory: {outputDir / splitName / 'images'}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AIST++ dataset to RTM Pose format')
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['pose_train', 'pose_val', 'pose_test'],
        help='Split to convert')
    parser.add_argument(
        '--splitsDir',
        type=str,
        default='splits',
        dest='splitsDir',
        help='Directory containing split files (default: splits)')
    parser.add_argument(
        '--framesDir',
        type=str,
        default='frames',
        dest='framesDir',
        help='Directory containing extracted frames (default: frames)')
    parser.add_argument(
        '--keypoints2dDir',
        type=str,
        default='keypoints2d',
        dest='keypoints2dDir',
        help='Directory containing keypoints2d annotations (default: keypoints2d)')
    parser.add_argument(
        '--outputDir',
        type=str,
        default='rtmpose_dataset',
        dest='outputDir',
        help='Output directory for RTM Pose dataset (default: rtmpose_dataset)')
    parser.add_argument(
        '--minConfidence',
        type=float,
        default=0.3,
        dest='minConfidence',
        help='Minimum confidence threshold for keypoints (default: 0.3)')
    parser.add_argument(
        '--videoName',
        type=str,
        default=None,
        dest='videoName',
        help='Convert single video (overrides --split)')
    
    args = parser.parse_args()
    
    if args.videoName:
        # Convert single video
        result = createRTMPoseDataset(
            args.videoName,
            args.framesDir,
            args.keypoints2dDir,
            args.outputDir,
            split='train',
            minConfidence=args.minConfidence
        )
        if result:
            print(f"✓ Converted {args.videoName}")
            print(f"  Images: {len(result['images'])}")
            print(f"  Annotations: {len(result['annotations'])}")
    else:
        # Convert entire split
        createRTMPoseDatasetFromSplit(
            args.split,
            args.splitsDir,
            args.framesDir,
            args.keypoints2dDir,
            args.outputDir,
            args.minConfidence
        )


if __name__ == '__main__':
    main()


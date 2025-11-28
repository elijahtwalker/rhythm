#!/usr/bin/env python3
"""
Visualize 2D keypoints from pose estimation pickle files.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# COCO pose skeleton connections (17 keypoints)
# Format: (start_idx, end_idx)
COCO_SKELETON = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (5, 6),   # left_shoulder -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def loadKeypoints(pklPath):
    """Load keypoints from pickle file."""
    with open(pklPath, 'rb') as f:
        data = pickle.load(f)
    return data['keypoints2d'], data.get('det_scores', None), data.get('timestamps', None)


def filterValidKeypoints(keypoints, minConfidence=0.3):
    """Filter out keypoints with low confidence."""
    # keypoints shape: (num_people, num_keypoints, 3)
    # Returns: list of valid person indices
    validPeople = []
    for i, person in enumerate(keypoints):
        # Check if person has enough valid keypoints
        confidences = person[:, 2]
        validKpCount = np.sum(confidences > minConfidence)
        if validKpCount >= 5:  # At least 5 keypoints with good confidence
            validPeople.append(i)
    return validPeople


def drawSkeleton(ax, keypoints, skeleton=COCO_SKELETON, confidenceThreshold=0.3, 
                  color='blue', linewidth=2, alpha=0.8):
    """Draw skeleton on matplotlib axes."""
    # keypoints shape: (num_keypoints, 3) where 3 is (x, y, confidence)
    
    # Draw connections
    for startIdx, endIdx in skeleton:
        startKp = keypoints[startIdx]
        endKp = keypoints[endIdx]
        
        # Only draw if both keypoints have sufficient confidence
        if startKp[2] > confidenceThreshold and endKp[2] > confidenceThreshold:
            ax.plot([startKp[0], endKp[0]], 
                   [startKp[1], endKp[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > confidenceThreshold:
            ax.scatter(kp[0], kp[1], color=color, s=50, alpha=alpha, zorder=5)


def visualizeFrame(keypointsFrame, frameIdx=0, maxPeople=5, minConfidence=0.3, 
                    figsize=(12, 8), savePath=None):
    """
    Visualize keypoints for a single frame.
    
    Args:
        keypointsFrame: Array of shape (num_people, num_keypoints, 3)
        frameIdx: Frame index for title
        maxPeople: Maximum number of people to visualize
        minConfidence: Minimum confidence threshold for keypoints
        figsize: Figure size
        savePath: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter valid people
    validPeople = filterValidKeypoints(keypointsFrame, minConfidence)
    numPeople = min(len(validPeople), maxPeople)
    
    if numPeople == 0:
        ax.text(0.5, 0.5, 'No valid detections', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'Frame {frameIdx} - No detections')
        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
        return fig
    
    # Get image dimensions from keypoints
    allX = []
    allY = []
    for personIdx in validPeople[:numPeople]:
        personKp = keypointsFrame[personIdx]
        validMask = personKp[:, 2] > minConfidence
        allX.extend(personKp[validMask, 0])
        allY.extend(personKp[validMask, 1])
    
    if len(allX) == 0:
        ax.text(0.5, 0.5, 'No valid keypoints', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'Frame {frameIdx} - No valid keypoints')
        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
        return fig
    
    # Set axis limits with padding
    xMin, xMax = min(allX), max(allX)
    yMin, yMax = min(allY), max(allY)
    padding = 50
    ax.set_xlim(xMin - padding, xMax + padding)
    ax.set_ylim(yMax + padding, yMin - padding)  # Invert y-axis (image coordinates)
    
    # Draw each person with different color
    colors = plt.cm.tab10(np.linspace(0, 1, numPeople))
    for idx, personIdx in enumerate(validPeople[:numPeople]):
        personKp = keypointsFrame[personIdx]
        drawSkeleton(ax, personKp, color=colors[idx], 
                     confidenceThreshold=minConfidence)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Frame {frameIdx} - {numPeople} person(s) detected')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if savePath:
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
    
    return fig


def visualizeSequence(keypoints, frameIndices=None, maxPeople=5, minConfidence=0.3,
                      figsize=(15, 10), savePath=None):
    """
    Visualize multiple frames from a sequence.
    
    Args:
        keypoints: Array of shape (num_frames, num_people, num_keypoints, 3)
        frameIndices: List of frame indices to visualize (default: first few frames)
        maxPeople: Maximum number of people to visualize per frame
        minConfidence: Minimum confidence threshold
        figsize: Figure size
        savePath: Path to save the figure (optional)
    """
    numFrames = len(keypoints)
    
    if frameIndices is None:
        # Show first 4 frames or all frames if less than 4
        frameIndices = list(range(min(4, numFrames)))
    
    nFrames = len(frameIndices)
    cols = 2
    rows = (nFrames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if nFrames == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, frameIdx in enumerate(frameIndices):
        if frameIdx >= numFrames:
            continue
        
        ax = axes[idx]
        keypointsFrame = keypoints[frameIdx]
        
        # Filter valid people
        validPeople = filterValidKeypoints(keypointsFrame, minConfidence)
        numPeople = min(len(validPeople), maxPeople)
        
        if numPeople == 0:
            ax.text(0.5, 0.5, 'No valid detections', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Frame {frameIdx}')
            continue
        
        # Get image dimensions
        allX, allY = [], []
        for personIdx in validPeople[:numPeople]:
            personKp = keypointsFrame[personIdx]
            validMask = personKp[:, 2] > minConfidence
            allX.extend(personKp[validMask, 0])
            allY.extend(personKp[validMask, 1])
        
        if len(allX) > 0:
            xMin, xMax = min(allX), max(allX)
            yMin, yMax = min(allY), max(allY)
            padding = 50
            ax.set_xlim(xMin - padding, xMax + padding)
            ax.set_ylim(yMax + padding, yMin - padding)
        
        # Draw each person
        colors = plt.cm.tab10(np.linspace(0, 1, numPeople))
        for pidx, personIdx in enumerate(validPeople[:numPeople]):
            personKp = keypointsFrame[personIdx]
            drawSkeleton(ax, personKp, color=colors[pidx], 
                         confidenceThreshold=minConfidence, linewidth=1.5)
        
        ax.set_title(f'Frame {frameIdx} ({numPeople} person(s))')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(nFrames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D keypoints from pickle files')
    parser.add_argument('pklFile', type=str, help='Path to pickle file')
    parser.add_argument('--frame', type=int, default=None, 
                       help='Frame index to visualize (default: show first few frames)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='Specific frame indices to visualize')
    parser.add_argument('--maxPeople', type=int, default=5,
                       dest='maxPeople',
                       help='Maximum number of people to visualize per frame (default: 5)')
    parser.add_argument('--minConfidence', type=float, default=0.3,
                       dest='minConfidence',
                       help='Minimum confidence threshold for keypoints (default: 0.3)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the visualization (optional)')
    parser.add_argument('--show', action='store_true',
                       help='Display the visualization (default: True if --save not specified)')
    
    args = parser.parse_args()
    
    # Load keypoints
    print(f"Loading keypoints from {args.pklFile}...")
    keypoints, detScores, timestamps = loadKeypoints(args.pklFile)
    print(f"Loaded {len(keypoints)} frames")
    print(f"Keypoints shape: {keypoints.shape}")
    
    # Determine what to visualize
    showPlot = args.show or (args.save is None)
    
    if args.frame is not None:
        # Visualize single frame
        if args.frame >= len(keypoints):
            print(f"Error: Frame {args.frame} out of range (max: {len(keypoints)-1})")
            return
        fig = visualizeFrame(keypoints[args.frame], frameIdx=args.frame,
                             maxPeople=args.maxPeople, 
                             minConfidence=args.minConfidence,
                             savePath=args.save)
    elif args.frames is not None:
        # Visualize specific frames
        fig = visualizeSequence(keypoints, frameIndices=args.frames,
                                maxPeople=args.maxPeople,
                                minConfidence=args.minConfidence,
                                savePath=args.save)
    else:
        # Visualize first few frames
        fig = visualizeSequence(keypoints, frameIndices=None,
                                maxPeople=args.maxPeople,
                                minConfidence=args.minConfidence,
                                savePath=args.save)
    
    if showPlot:
        plt.show()
    else:
        plt.close(fig)
        print(f"Visualization saved to {args.save}")


if __name__ == '__main__':
    main()


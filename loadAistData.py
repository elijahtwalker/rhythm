#!/usr/bin/env python3
"""
Load AIST++ dataset using the proper API structure.
Supports loading keypoints2d, keypoints3d, motions, and camera data
with train/validation/test splits.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse


class AISTDataLoader:
    """
    Data loader for AIST++ dataset following the official API structure.
    
    Expected directory structure:
    <ANNOTATIONS_DIR>
    ├── motions/
    ├── keypoints2d/
    ├── keypoints3d/
    ├── splits/
    │   ├── pose_train.txt
    │   ├── pose_val.txt
    │   ├── pose_test.txt
    │   └── ...
    ├── cameras/
    └── ignore_list.txt
    """
    
    def __init__(self, annoDir: str):
        """
        Initialize the data loader.
        
        Args:
            annoDir: Path to annotations directory
        """
        self.annoDir = Path(annoDir)
        self.keypoints2dDir = self.annoDir / 'keypoints2d'
        self.keypoints3dDir = self.annoDir / 'keypoints3d'
        self.motionsDir = self.annoDir / 'motions'
        self.camerasDir = self.annoDir / 'cameras'
        self.splitsDir = self.annoDir / 'splits'
        
        # Check which data types are available
        self.hasKeypoints2d = self.keypoints2dDir.exists()
        self.hasKeypoints3d = self.keypoints3dDir.exists()
        self.hasMotions = self.motionsDir.exists()
        self.hasCameras = self.camerasDir.exists()
        
        print(f"Data availability:")
        print(f"  - keypoints2d: {self.hasKeypoints2d}")
        print(f"  - keypoints3d: {self.hasKeypoints3d}")
        print(f"  - motions: {self.hasMotions}")
        print(f"  - cameras: {self.hasCameras}")
    
    def loadSplitNames(self, split: str) -> List[str]:
        """
        Load video names from a split file.
        
        Args:
            split: Split name ('train', 'val', 'test', 'pose_train', 'pose_val', etc.)
        
        Returns:
            List of video names (without extension)
        """
        # Handle different split naming conventions
        splitFile = None
        if split in ['train', 'pose_train']:
            splitFile = self.splitsDir / 'pose_train.txt'
        elif split in ['val', 'validation', 'pose_val']:
            splitFile = self.splitsDir / 'pose_val.txt'
        elif split in ['test', 'pose_test']:
            splitFile = self.splitsDir / 'pose_test.txt'
        elif split == 'all':
            splitFile = self.splitsDir / 'all.txt'
        else:
            # Try direct filename
            splitFile = self.splitsDir / f'{split}.txt'
        
        if not splitFile.exists():
            raise FileNotFoundError(f"Split file not found: {splitFile}")
        
        with open(splitFile, 'r') as f:
            videoNames = [line.strip() for line in f if line.strip()]
        
        return videoNames
    
    def loadKeypoints2d(self, videoName: str) -> Optional[Dict]:
        """
        Load 2D keypoints for a video.
        
        Args:
            videoName: Video name (without extension)
        
        Returns:
            Dictionary with keys: 'keypoints2d', 'det_scores', 'timestamps'
            Returns None if file doesn't exist
        """
        if not self.hasKeypoints2d:
            return None
        
        pklPath = self.keypoints2dDir / f'{videoName}.pkl'
        if not pklPath.exists():
            return None
        
        with open(pklPath, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def loadKeypoints3d(self, videoName: str) -> Optional[Dict]:
        """
        Load 3D keypoints for a video.
        
        Args:
            videoName: Video name (without extension)
        
        Returns:
            Dictionary with 3D keypoints data
            Returns None if file doesn't exist
        """
        if not self.hasKeypoints3d:
            return None
        
        pklPath = self.keypoints3dDir / f'{videoName}.pkl'
        if not pklPath.exists():
            return None
        
        with open(pklPath, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def loadMotion(self, videoName: str) -> Optional[Dict]:
        """
        Load SMPL motion data for a video.
        
        Args:
            videoName: Video name (without extension)
        
        Returns:
            Dictionary with SMPL motion data
            Returns None if file doesn't exist
        """
        if not self.hasMotions:
            return None
        
        pklPath = self.motionsDir / f'{videoName}.pkl'
        if not pklPath.exists():
            return None
        
        with open(pklPath, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def loadCamera(self, videoName: str) -> Optional[Dict]:
        """
        Load camera parameters for a video.
        
        Args:
            videoName: Video name (without extension)
        
        Returns:
            Dictionary with camera parameters
            Returns None if file doesn't exist
        """
        if not self.hasCameras:
            return None
        
        # Camera files might be named differently, try common patterns
        possiblePaths = [
            self.camerasDir / f'{videoName}.json',
            self.camerasDir / f'{videoName}.pkl',
            self.camerasDir / f'{videoName}.npz',
        ]
        
        for path in possiblePaths:
            if path.exists():
                if path.suffix == '.json':
                    import json
                    with open(path, 'r') as f:
                        return json.load(f)
                elif path.suffix == '.pkl':
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                elif path.suffix == '.npz':
                    return dict(np.load(path))
        
        return None
    
    def loadVideoData(self, videoName: str, 
                       loadKeypoints2d: bool = True,
                       loadKeypoints3d: bool = True,
                       loadMotion: bool = True,
                       loadCamera: bool = True) -> Dict:
        """
        Load all available data for a video.
        
        Args:
            videoName: Video name (without extension)
            loadKeypoints2d: Whether to load 2D keypoints
            loadKeypoints3d: Whether to load 3D keypoints
            loadMotion: Whether to load motion data
            loadCamera: Whether to load camera data
        
        Returns:
            Dictionary containing all loaded data
        """
        data = {'video_name': videoName}
        
        if loadKeypoints2d:
            data['keypoints2d'] = self.loadKeypoints2d(videoName)
        
        if loadKeypoints3d:
            data['keypoints3d'] = self.loadKeypoints3d(videoName)
        
        if loadMotion:
            data['motion'] = self.loadMotion(videoName)
        
        if loadCamera:
            data['camera'] = self.loadCamera(videoName)
        
        return data
    
    def loadSplit(self, split: str,
                  loadKeypoints2d: bool = True,
                  loadKeypoints3d: bool = True,
                  loadMotion: bool = True,
                  loadCamera: bool = True,
                  verbose: bool = True) -> List[Dict]:
        """
        Load all videos from a split.
        
        Args:
            split: Split name ('train', 'val', 'test', etc.)
            loadKeypoints2d: Whether to load 2D keypoints
            loadKeypoints3d: Whether to load 3D keypoints
            loadMotion: Whether to load motion data
            loadCamera: Whether to load camera data
            verbose: Whether to print progress
        
        Returns:
            List of dictionaries, one per video
        """
        videoNames = self.loadSplitNames(split)
        dataList = []
        
        for i, videoName in enumerate(videoNames):
            if verbose and (i + 1) % 100 == 0:
                print(f"Loading {i + 1}/{len(videoNames)}...")
            
            videoData = self.loadVideoData(
                videoName,
                loadKeypoints2d=loadKeypoints2d,
                loadKeypoints3d=loadKeypoints3d,
                loadMotion=loadMotion,
                loadCamera=loadCamera
            )
            dataList.append(videoData)
        
        if verbose:
            print(f"Loaded {len(dataList)} videos from {split} split")
        
        return dataList
    
    def getStatistics(self, split: Optional[str] = None) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            split: Optional split name to get stats for specific split
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if split:
            videoNames = self.loadSplitNames(split)
            stats['split'] = split
            stats['numVideos'] = len(videoNames)
        else:
            # Get stats for all splits
            for splitName in ['pose_train', 'pose_val', 'pose_test']:
                try:
                    videoNames = self.loadSplitNames(splitName)
                    stats[splitName] = len(videoNames)
                except FileNotFoundError:
                    pass
        
        # Count available data files
        if self.hasKeypoints2d:
            stats['keypoints2dFiles'] = len(list(self.keypoints2dDir.glob('*.pkl')))
        
        if self.hasKeypoints3d:
            stats['keypoints3dFiles'] = len(list(self.keypoints3dDir.glob('*.pkl')))
        
        if self.hasMotions:
            stats['motionFiles'] = len(list(self.motionsDir.glob('*.pkl')))
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Load AIST++ dataset using the proper API structure')
    parser.add_argument(
        '--annoDir',
        type=str,
        default='.',
        dest='annoDir',
        help='Path to annotations directory (default: current directory)')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'pose_train', 'pose_val', 'pose_test', 'all'],
        help='Which split to load (default: train)')
    parser.add_argument(
        '--videoName',
        type=str,
        default=None,
        dest='videoName',
        help='Load specific video by name (overrides --split)')
    parser.add_argument(
        '--no-keypoints2d',
        action='store_true',
        help='Skip loading 2D keypoints')
    parser.add_argument(
        '--no-keypoints3d',
        action='store_true',
        help='Skip loading 3D keypoints')
    parser.add_argument(
        '--no-motion',
        action='store_true',
        help='Skip loading motion data')
    parser.add_argument(
        '--no-camera',
        action='store_true',
        help='Skip loading camera data')
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print dataset statistics')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = AISTDataLoader(args.annoDir)
    
    # Print statistics if requested
    if args.stats:
        stats = loader.getStatistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Load data
    if args.videoName:
        # Load specific video
        print(f"Loading video: {args.videoName}")
        data = loader.loadVideoData(
            args.videoName,
            loadKeypoints2d=not args.no_keypoints2d,
            loadKeypoints3d=not args.no_keypoints3d,
            loadMotion=not args.no_motion,
            loadCamera=not args.no_camera
        )
        print(f"\nLoaded data for {args.videoName}:")
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    else:
        # Load split
        print(f"Loading {args.split} split...")
        dataList = loader.loadSplit(
            args.split,
            loadKeypoints2d=not args.no_keypoints2d,
            loadKeypoints3d=not args.no_keypoints3d,
            loadMotion=not args.no_motion,
            loadCamera=not args.no_camera
        )
        print(f"\nLoaded {len(dataList)} videos from {args.split} split")
        
        # Show sample
        if len(dataList) > 0:
            print(f"\nSample video: {dataList[0]['video_name']}")
            for key, value in dataList[0].items():
                if key != 'video_name' and value is not None:
                    if isinstance(value, dict):
                        print(f"  {key}: {list(value.keys())}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value).__name__}")


if __name__ == '__main__':
    main()


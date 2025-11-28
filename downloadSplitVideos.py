#!/usr/bin/env python3
"""
Download and organize AIST++ videos by splits.
Downloads videos listed in split files and organizes them into split directories.
Handles cAll videos by converting to available camera-specific versions.
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from functools import partial
import multiprocessing


SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'


def getAvailableVideoNames(videoName):
    """
    Convert cAll video names to available camera-specific names.
    Returns list of possible camera-specific video names to try.
    """
    if 'cAll' not in videoName:
        return [videoName]
    
    # Try cameras c01 through c10
    availableNames = []
    for camNum in range(1, 11):
        camName = f'c{camNum:02d}'
        availableNames.append(videoName.replace('cAll', camName))
    
    return availableNames


def downloadVideo(videoName, downloadFolder, splitName=None, showProgress=True):
    """Download a single video with progress reporting."""
    if splitName:
        splitDir = Path(downloadFolder) / splitName
        splitDir.mkdir(parents=True, exist_ok=True)
        savePath = splitDir / f"{videoName}.mp4"
    else:
        savePath = Path(downloadFolder) / f"{videoName}.mp4"
    
    # Skip if already exists and has reasonable size
    if savePath.exists():
        fileSize = savePath.stat().st_size
        if fileSize > 1024 * 1024:  # At least 1MB
            fileSizeMB = fileSize / (1024 * 1024)
            if showProgress:
                print(f"  ✓ {videoName}.mp4 already exists ({fileSizeMB:.1f} MB)")
            return True, "exists"
    
    # Get available video names to try (handles cAll → c01, c02, etc.)
    availableNames = getAvailableVideoNames(videoName)
    actualVideoName = None
    
    # Try each available camera version
    for candidateName in availableNames:
        videoUrl = f"{SOURCE_URL}{candidateName}.mp4"
        
        if showProgress and len(availableNames) > 1:
            print(f"  Trying {candidateName}.mp4 (from {videoName})...", end='', flush=True)
        elif showProgress:
            print(f"  Downloading {videoName}.mp4...", end='', flush=True)
        
        maxRetries = 3
        success = False
        
        for attempt in range(maxRetries):
            try:
                def reporthook(count, blockSize, totalSize):
                    if totalSize > 0 and showProgress:
                        percent = min(100, int(count * blockSize * 100 / totalSize))
                        downloadedMB = (count * blockSize) / (1024 * 1024)
                        totalMB = totalSize / (1024 * 1024)
                        displayName = candidateName if len(availableNames) > 1 else videoName
                        print(f"\r  Downloading {displayName}.mp4... {percent}% ({downloadedMB:.1f}/{totalMB:.1f} MB)", 
                              end='', flush=True)
                
                urllib.request.urlretrieve(videoUrl, str(savePath), reporthook=reporthook)
                
                if showProgress:
                    print()  # New line after progress
                
                if savePath.exists() and savePath.stat().st_size > 0:
                    actualVideoName = candidateName
                    success = True
                    break
                else:
                    if savePath.exists():
                        savePath.unlink()
                    raise Exception("Downloaded file is empty")
                    
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # This camera version doesn't exist, try next
                    if showProgress:
                        print(f"\r  {candidateName}.mp4 not found (404), trying next camera...", end='', flush=True)
                    break  # Try next camera
                elif attempt < maxRetries - 1:
                    if showProgress:
                        print(f"\r  Retrying {candidateName}.mp4... (HTTP {e.code})", end='', flush=True)
                    continue
                else:
                    if showProgress:
                        print(f"\r  ✗ Failed: {candidateName}.mp4 - HTTP {e.code}")
                    break  # Try next camera
            except Exception as e:
                if attempt < maxRetries - 1:
                    if showProgress:
                        print(f"\r  Retrying {candidateName}.mp4... (attempt {attempt + 2}/{maxRetries})", end='', flush=True)
                    continue
                else:
                    if showProgress:
                        print(f"\r  ✗ Failed: {candidateName}.mp4 - {str(e)}")
                    break  # Try next camera
        
        if success:
            break
    
    if success and actualVideoName:
        fileSizeMB = savePath.stat().st_size / (1024 * 1024)
        if showProgress:
            if actualVideoName != videoName:
                print(f"  ✓ Downloaded {actualVideoName}.mp4 as {videoName}.mp4 ({fileSizeMB:.1f} MB)")
            else:
                print(f"  ✓ Downloaded {videoName}.mp4 ({fileSizeMB:.1f} MB)")
        return True, "downloaded"
    else:
        if showProgress:
            if len(availableNames) > 1:
                print(f"\r  ✗ Failed: {videoName}.mp4 - No camera version available (tried c01-c10)")
            else:
                print(f"\r  ✗ Failed: {videoName}.mp4 - Download failed")
        return False, "Download failed"


def downloadVideosForSplit(splitFile, downloadFolder, numProcesses=1):
    """Download all videos listed in a split file."""
    splitPath = Path(splitFile)
    if not splitPath.exists():
        print(f"Error: Split file not found: {splitFile}")
        return
    
    # Get split name from filename
    splitName = splitPath.stem  # e.g., 'pose_train' from 'pose_train.txt'
    
    # Read video names
    with open(splitPath, 'r') as f:
        videoNames = [line.strip() for line in f if line.strip()]
    
    print(f"\nDownloading {len(videoNames)} videos for {splitName}...")
    if any('cAll' in name for name in videoNames):
        print("  Note: Videos with 'cAll' will be downloaded as camera-specific versions (c01, c02, etc.)")
    
    successful = 0
    failed = 0
    skipped = 0
    failedVideos = []
    
    if numProcesses > 1:
        # Multiprocessing (progress shown per video)
        downloadFunc = partial(downloadVideo, 
                              downloadFolder=downloadFolder, 
                              splitName=splitName,
                              showProgress=True)
        pool = multiprocessing.Pool(processes=numProcesses)
        results = pool.map(downloadFunc, videoNames)
        pool.close()
        pool.join()
        
        for videoName, (success, message) in zip(videoNames, results):
            if success:
                if message == "exists":
                    skipped += 1
                else:
                    successful += 1
            else:
                failed += 1
                failedVideos.append(videoName)
    else:
        # Single-threaded with detailed progress
        for i, videoName in enumerate(videoNames):
            print(f"\n[{i + 1}/{len(videoNames)}] ", end='', flush=True)
            success, message = downloadVideo(videoName, downloadFolder, splitName, showProgress=True)
            
            if success:
                if message == "exists":
                    skipped += 1
                else:
                    successful += 1
            else:
                failed += 1
                failedVideos.append(videoName)
            
            # Show summary every 10 videos
            if (i + 1) % 10 == 0:
                print(f"\n  Summary: {i + 1}/{len(videoNames)} processed "
                     f"(✓ {successful}, ✗ {failed}, ⊘ {skipped})")
    
    print(f"\n{splitName} complete:")
    print(f"  Successful: {successful}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        failedFile = Path(downloadFolder) / f"{splitName}_failed.txt"
        with open(failedFile, 'w') as f:
            for videoName in failedVideos:
                f.write(f"{videoName}\n")
        print(f"  Failed videos list saved to: {failedFile}")
        
        # Check if most failures are 404s
        if failed > len(videoNames) * 0.5:  # More than 50% failed
            print(f"\n⚠ Warning: {failed} videos failed to download.")
            print(f"  Videos with 'cAll' are automatically converted to camera-specific versions.")
            print(f"  If videos still fail, they may not be available in the official video list.")
            print(f"  Check: https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Download AIST++ videos organized by splits')
    parser.add_argument(
        '--splitsDir',
        type=str,
        default='splits',
        dest='splitsDir',
        help='Directory containing split files (default: splits)')
    parser.add_argument(
        '--downloadFolder',
        type=str,
        default='videos',
        dest='downloadFolder',
        help='Directory to download videos (default: videos)')
    parser.add_argument(
        '--split',
        type=str,
        choices=['pose_train', 'pose_val', 'pose_test', 'all'],
        default='all',
        help='Which split to download (default: all)')
    parser.add_argument(
        '--numProcesses',
        type=int,
        default=4,
        dest='numProcesses',
        help='Number of parallel download processes (default: 4)')
    
    args = parser.parse_args()
    
    # Check for terms of use agreement
    ans = input(
        "Before running this script, please make sure you have read the <Terms of Use> "
        "of AIST Dance Video Database at here: \n"
        "\n"
        "https://aistdancedb.ongaaccel.jp/terms_of_use/\n"
        "\n"
        "Do you agree with the <Terms of Use>? [Y/N] "
    )
    if ans not in ["Yes", "YES", "yes", "Y", "y"]:
        print("Program exit. Please first acknowledge the <Terms of Use>.")
        sys.exit(1)
    
    # Create download folder
    os.makedirs(args.downloadFolder, exist_ok=True)
    
    # Determine which splits to download
    if args.split == 'all':
        splits = ['pose_train', 'pose_val', 'pose_test']
    else:
        splits = [args.split]
    
    # Download videos for each split
    for splitName in splits:
        splitFile = Path(args.splitsDir) / f"{splitName}.txt"
        if splitFile.exists():
            downloadVideosForSplit(
                splitFile, 
                args.downloadFolder, 
                numProcesses=args.numProcesses
            )
        else:
            print(f"Warning: Split file not found: {splitFile}")
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nVideos are organized in: {args.downloadFolder}/")
    for splitName in splits:
        splitDir = Path(args.downloadFolder) / splitName
        if splitDir.exists():
            videoCount = len(list(splitDir.glob("*.mp4")))
            print(f"  {splitName}: {videoCount} videos")


if __name__ == '__main__':
    main()

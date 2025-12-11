from pathlib import Path

SKIP_SPLITS = {'pose_train_sample'}


def checkSplitStatus(splitsDir, videoDir, keypoints2dDir):
    """Check status of data for each split."""
    splitsDir = Path(splitsDir)
    videoDir = Path(videoDir)
    keypoints2dDir = Path(keypoints2dDir) if keypoints2dDir else None
    
    results = {}
    
    for splitFile in splitsDir.glob("*.txt"):
        splitName = splitFile.stem
        if splitName in SKIP_SPLITS:
            continue
        
        with open(splitFile, 'r') as f:
            videoNames = [line.strip() for line in f if line.strip()]
        
        stats = {
            'total': len(videoNames),
            'videos': 0,
            'keypoints2d': 0,
            'videosInSplitDir': 0,
            'videoNames': set(),
            'downloadedVideos': set(),
            'keypointVideos': set()
        }
        
        # check if videos exist
        for videoName in videoNames:
            videoFound = False
            stats['videoNames'].add(videoName)

            if (videoDir / f"{videoName}.mp4").exists():
                stats['videos'] += 1
                videoFound = True
                stats['downloadedVideos'].add(videoName)
            
            splitVideoDir = videoDir / splitName
            if splitVideoDir.exists() and (splitVideoDir / f"{videoName}.mp4").exists():
                stats['videosInSplitDir'] += 1
                if not videoFound:
                    stats['videos'] += 1
                    videoFound = True
                    stats['downloadedVideos'].add(videoName)
            
            # check keypoints2d
            if keypoints2dDir and (keypoints2dDir / f"{videoName}.pkl").exists():
                stats['keypoints2d'] += 1
                stats['keypointVideos'].add(videoName)
        
        results[splitName] = stats
    
    return results


def printStatusReport(results):
    print("\n" + "="*70)
    print("AIST++ Dataset Status Report")
    print("="*70)
    
    uniqueVideos = set()
    uniqueDownloaded = set()
    uniqueKeypoints = set()
    
    for splitName, stats in sorted(results.items()):
        uniqueVideos.update(stats.get('videoNames', []))
        uniqueDownloaded.update(stats.get('downloadedVideos', []))
        uniqueKeypoints.update(stats.get('keypointVideos', []))
        
        print(f"\n{splitName}:")
        print(f"  Total videos in split: {stats['total']}")
        print(f"  Videos downloaded: {stats['videos']} ({100*stats['videos']/stats['total']:.1f}%)")
        print(f"  Videos in split directory: {stats['videosInSplitDir']} ({100*stats['videosInSplitDir']/stats['total']:.1f}%)")
        if stats.get('keypoints2d', 0) > 0:
            print(f"  Keypoints2D available: {stats['keypoints2d']} ({100*stats['keypoints2d']/stats['total']:.1f}%)")
    
    totalVideos = len(uniqueVideos)
    totalDownloaded = len(uniqueDownloaded)
    totalKeypoints = len(uniqueKeypoints)
    
    print("\n" + "-"*70)
    print("Summary:")
    print(f"  Total unique videos: {totalVideos}")
    if totalVideos > 0:
        print(f"  Videos downloaded: {totalDownloaded} ({100*totalDownloaded/totalVideos:.1f}%)")
        print(f"  Keypoints2D available: {totalKeypoints} ({100*totalKeypoints/totalVideos:.1f}%)")
    else:
        print("  Videos downloaded: 0 (0.0%)")
        print("  Keypoints2D available: 0 (0.0%)")
    print("="*70)
    
    # Recommendations
    print("\nRecommendations:")
    if totalVideos > 0 and totalDownloaded < totalVideos:
        missing = totalVideos - totalDownloaded
        print(f"  - Download {missing} missing videos:")
        print(f"    python3 downloadSplitVideos.py --split all --numProcesses 4")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check AIST++ dataset download status')
    parser.add_argument(
        '--splitsDir',
        type=str,
        default='splits',
        dest='splitsDir',
        help='Directory containing split files (default: splits)')
    parser.add_argument(
        '--videoDir',
        type=str,
        default='videos',
        dest='videoDir',
        help='Directory containing videos (default: videos)')
    parser.add_argument(
        '--keypoints2dDir',
        type=str,
        default='keypoints2d',
        dest='keypoints2dDir',
        help='Directory containing keypoints2d (default: keypoints2d)')
    
    args = parser.parse_args()

    if not Path(args.splitsDir).exists():
        print(f"Error: Splits directory not found: {args.splitsDir}")
        return
    
    if not Path(args.videoDir).exists():
        print(f"Note: Video directory not found: {args.videoDir}")
        print("  Run: python3 downloadSplitVideos.py --split all")

    results = checkSplitStatus(args.splitsDir, args.videoDir, args.keypoints2dDir)
    printStatusReport(results)


if __name__ == '__main__':
    main()


import pickle
import json
import numpy as np
import os
from pathlib import Path
import sys

cocoKeypointNames = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

mpToCocoMap = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}


def loadPkl(filepath):
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data


def loadJson(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def estimateResolution(gtData, mpDataDict, frameNames):
    ratiosX = []
    ratiosY = []
    numFrames = gtData["keypoints2d"].shape[1]
    for frameIdx, frameName in enumerate(frameNames):
        if frameIdx >= numFrames:
            break
        mpLandmarks = mpDataDict.get(frameName)
        if not mpLandmarks:
            continue
        gtFrame = gtData["keypoints2d"][:, frameIdx, :, :]
        for mpIdx, cocoIdx in mpToCocoMap.items():
            if mpIdx >= len(mpLandmarks):
                continue
            mpPt = mpLandmarks[mpIdx]
            mpX, mpY = mpPt["x"], mpPt["y"]
            mpVis = mpPt.get("visibility", 0)
            if mpVis < 0.5:
                continue
            for personIdx in range(gtFrame.shape[0]):
                gtPt = gtFrame[personIdx, cocoIdx]
                gtX, gtY, gtConf = gtPt
                if gtConf > 0.5 and gtX > 1 and gtY > 1:
                    if mpX > 0.01:
                        ratiosX.append(gtX / mpX)
                    if mpY > 0.01:
                        ratiosY.append(gtY / mpY)
    if not ratiosX or not ratiosY:
        print("Warning: Could not auto-calibrate resolution. Defaulting to 1920x1080.")
        return 1920, 1080
    return np.median(ratiosX), np.median(ratiosY)


def calculatePdjForFrame(gtPersons, mpLandmarks, width, height, fraction=0.05):
    if not mpLandmarks:
        return 0.0, [False] * 17
    mpKptsCoco = np.zeros((17, 2))
    for mpIdx, cocoIdx in mpToCocoMap.items():
        point = mpLandmarks[mpIdx]
        mpKptsCoco[cocoIdx] = [point["x"] * width, point["y"] * height]
    bestPersonPdjAvg = -1.0
    bestPersonDetections = [False] * 17
    numPersons = gtPersons.shape[0]
    for personIdx in range(numPersons):
        gtKpts = gtPersons[personIdx]
        validGt = gtKpts[gtKpts[:, 2] > 0.1]
        if len(validGt) < 2:
            continue
        minX, minY = np.min(validGt[:, :2], axis=0)
        maxX, maxY = np.max(validGt[:, :2], axis=0)
        widthBox = maxX - minX
        heightBox = maxY - minY
        diagonal = np.sqrt(widthBox ** 2 + heightBox ** 2)
        if diagonal < 1.0:
            diagonal = 1.0
        thresholdDistance = fraction * diagonal
        currentDetections = []
        validKptCount = 0
        detectedCount = 0
        for keypointIndex in range(17):
            gtPt = gtKpts[keypointIndex]
            gtX, gtY, gtConf = gtPt
            if gtConf < 0.1:
                currentDetections.append(False)
                continue
            mpX, mpY = mpKptsCoco[keypointIndex]
            distance = np.sqrt((gtX - mpX) ** 2 + (gtY - mpY) ** 2)
            isDetected = distance < thresholdDistance
            currentDetections.append(isDetected)
            validKptCount += 1
            if isDetected:
                detectedCount += 1
        if validKptCount > 0:
            pdjScore = detectedCount / validKptCount
        else:
            pdjScore = 0.0
        if pdjScore > bestPersonPdjAvg:
            bestPersonPdjAvg = pdjScore
            bestPersonDetections = currentDetections
    if bestPersonPdjAvg == -1.0:
        return 0.0, [False] * 17
    return bestPersonPdjAvg, bestPersonDetections


def processFilePair(pklPath, jsonPath, fraction=0.05):
    print(f"Processing Pair: {pklPath.name}")
    gtData = loadPkl(pklPath)
    mpData = loadJson(jsonPath)
    mpFrameNames = sorted(mpData.keys())
    width, height = estimateResolution(gtData, mpData, mpFrameNames)
    results = []
    gtKptsAll = gtData["keypoints2d"]
    for frameIdx, frameName in enumerate(mpFrameNames):
        if frameIdx >= gtKptsAll.shape[1]:
            break
        mpLandmarks = mpData.get(frameName)
        gtPersons = gtKptsAll[:, frameIdx, :, :]
        pdjAverage, detections = calculatePdjForFrame(
            gtPersons, mpLandmarks, width, height, fraction
        )
        results.append(
            {
                "frame": frameName,
                "pdj": pdjAverage,
                "detections": detections,
            }
        )
    return results


def main():
    rootDir = Path(".")
    keypoints2dDir = rootDir / "keypoints2d"
    mediapipeDir = rootDir / "keypoints_mediapipe"
    if not keypoints2dDir.exists() or not mediapipeDir.exists():
        print("Error: keypoints directories not found.")
        return
    pklFiles = list(keypoints2dDir.glob("*.pkl"))
    totalFrames = 0
    globalPdjSum = 0.0
    totalDetectedPerKpt = {name: 0 for name in cocoKeypointNames}
    fraction = 0.05
    print(f"Calculating PDJ with threshold = {fraction} * bounding_box_diagonal\n")
    for pklFile in pklFiles:
        baseName = pklFile.stem
        jsonFile = mediapipeDir / f"{baseName}_keypoints.json"
        if not jsonFile.exists():
            continue
        fileResults = processFilePair(pklFile, jsonFile, fraction)
        if fileResults:
            print(f"  Frame 0 PDJ: {fileResults[0]['pdj']:.4f}")
        for result in fileResults:
            totalFrames += 1
            globalPdjSum += result["pdj"]
            for index, isDetected in enumerate(result["detections"]):
                if isDetected:
                    keypointName = cocoKeypointNames[index]
                    totalDetectedPerKpt[keypointName] += 1
    print(f"\n--- PDJ Summary (Threshold={fraction}*Diagonal) ---")
    print(f"Total Frames Processed: {totalFrames}")
    if totalFrames > 0:
        averageGlobalPdj = globalPdjSum / totalFrames
        print(f"Global Average PDJ: {averageGlobalPdj:.4f}")
        print("\nPer Keypoint Statistics:")
        print(f"{'Keypoint':<20} {'Detections':<12} {'% Detected':<12}")
        print("-" * 46)
        for keypointName in cocoKeypointNames:
            count = totalDetectedPerKpt[keypointName]
            percentage = (count / totalFrames) * 100
            print(f"{keypointName:<20} {count:<12} {percentage:<12.1f}")


if __name__ == "__main__":
    main()
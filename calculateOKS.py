import pickle
import json
import numpy as np
import os
from pathlib import Path
import sys

cocoSigmas = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ]
)

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
    estW = np.median(ratiosX)
    estH = np.median(ratiosY)
    return estW, estH


def calculateOksForFrame(gtPersons, mpLandmarks, width, height):
    if not mpLandmarks:
        return [0.0] * 17, 0.0
    mpKptsCoco = np.zeros((17, 2))
    mpVisCoco = np.zeros(17)
    for mpIdx, cocoIdx in mpToCocoMap.items():
        point = mpLandmarks[mpIdx]
        mpKptsCoco[cocoIdx] = [point["x"] * width, point["y"] * height]
        mpVisCoco[cocoIdx] = point.get("visibility", 0)
    bestPersonOksAvg = -1
    bestPersonOksPerKpt = [0.0] * 17
    numPersons = gtPersons.shape[0]
    for personIdx in range(numPersons):
        gtKpts = gtPersons[personIdx]
        validGt = gtKpts[gtKpts[:, 2] > 0.1]
        if len(validGt) < 3:
            continue
        minX, minY = np.min(validGt[:, :2], axis=0)
        maxX, maxY = np.max(validGt[:, :2], axis=0)
        area = (maxX - minX) * (maxY - minY)
        scale = np.sqrt(area)
        if scale < 1:
            scale = 1.0
        currentOksList = []
        validKptCount = 0
        sumOks = 0.0
        for kIndex in range(17):
            gtPt = gtKpts[kIndex]
            gtX, gtY, gtConf = gtPt
            if gtConf < 0.1:
                currentOksList.append(0.0)
                continue
            mpX, mpY = mpKptsCoco[kIndex]
            distanceSquared = (gtX - mpX) ** 2 + (gtY - mpY) ** 2
            sigma = cocoSigmas[kIndex]
            oksValue = np.exp(-distanceSquared / (2 * (scale ** 2) * (sigma ** 2)))
            currentOksList.append(oksValue)
            sumOks += oksValue
            validKptCount += 1
        if validKptCount > 0:
            averageOks = sumOks / validKptCount
        else:
            averageOks = 0.0
        if averageOks > bestPersonOksAvg:
            bestPersonOksAvg = averageOks
            bestPersonOksPerKpt = currentOksList
    if bestPersonOksAvg == -1:
        return [0.0] * 17, 0.0
    return bestPersonOksPerKpt, bestPersonOksAvg


def processFilePair(pklPath, jsonPath, threshold=0.5):
    print(f"Processing Pair:\n  GT: {pklPath}\n  MP: {jsonPath}")
    gtData = loadPkl(pklPath)
    mpData = loadJson(jsonPath)
    mpFrameNames = sorted(mpData.keys())
    width, height = estimateResolution(gtData, mpData, mpFrameNames)
    print(f"  Estimated Resolution: {width:.1f} x {height:.1f}")
    results = []
    gtKptsAll = gtData["keypoints2d"]
    numFrames = gtKptsAll.shape[1]
    for frameIdx, frameName in enumerate(mpFrameNames):
        if frameIdx >= numFrames:
            break
        mpLandmarks = mpData.get(frameName)
        gtPersons = gtKptsAll[:, frameIdx, :, :]
        oksValues, averageOks = calculateOksForFrame(
            gtPersons, mpLandmarks, width, height
        )
        frameResult = {
            "frame": frameName,
            "average_oks": averageOks,
            "keypoints": {},
        }
        for index, value in enumerate(oksValues):
            keypointName = cocoKeypointNames[index]
            detected = value > threshold
            frameResult["keypoints"][keypointName] = {
                "oks": value,
                "detected": detected,
            }
        results.append(frameResult)
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
    totalDetectedPerKpt = {name: 0 for name in cocoKeypointNames}
    totalOksSumPerKpt = {name: 0.0 for name in cocoKeypointNames}
    globalOksSum = 0.0
    for pklFile in pklFiles:
        baseName = pklFile.stem
        jsonFile = mediapipeDir / f"{baseName}_keypoints.json"
        if not jsonFile.exists():
            print(f"Skipping {pklFile.name} (No matching JSON)")
            continue
        fileResults = processFilePair(pklFile, jsonFile, threshold=0.5)
        for result in fileResults:
            totalFrames += 1
            globalOksSum += result["average_oks"]
            for keypointName, data in result["keypoints"].items():
                totalOksSumPerKpt[keypointName] += data["oks"]
                if data["detected"]:
                    totalDetectedPerKpt[keypointName] += 1
        if fileResults:
            print(f"  Frame 0 Avg OKS: {fileResults[0]['average_oks']:.4f}")
    print("\n--- Summary (Threshold=0.5) ---")
    print(f"Total Frames Processed: {totalFrames}")
    if totalFrames > 0:
        averageGlobalOks = globalOksSum / totalFrames
        print(f"Global Average OKS: {averageGlobalOks:.4f}")
        print("\nPer Keypoint Statistics:")
        print(f"{'Keypoint':<20} {'Detections':<12} {'% Detected':<12} {'Avg OKS'}")
        print("-" * 60)
        for keypointName in cocoKeypointNames:
            count = totalDetectedPerKpt[keypointName]
            percentage = (count / totalFrames) * 100
            averageKeypointOks = totalOksSumPerKpt[keypointName] / totalFrames
            print(
                f"{keypointName:<20} {count:<12} {percentage:<12.1f} {averageKeypointOks:.4f}"
            )


if __name__ == "__main__":
    main()
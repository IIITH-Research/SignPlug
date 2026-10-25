# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options
    )
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("Inputs/Tpose-bodyPose.png")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)


# --- EXTRACTED CELL ---


PoseWorldLandmarks = detection_result.pose_world_landmarks


# --- EXTRACTED CELL ---


# STEP 5: Process the detection result. In this case, visualize it.
type(PoseWorldLandmarks)
print(PoseWorldLandmarks)
print(len(PoseWorldLandmarks[0]))


# --- EXTRACTED CELL ---



# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("Inputs/Tpose-hands.png")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)


# --- EXTRACTED CELL ---

RightHandWorldLandmarks = detection_result.hand_world_landmarks[0]


# --- EXTRACTED CELL ---



# STEP 5: Process the classification result. In this case, visualize it.
detection_result.handedness


# --- EXTRACTED CELL ---

LeftHandWorldLandmarks = detection_result.hand_world_landmarks[1]
detection_result.handedness[0][0].category_name      
print(len(RightHandWorldLandmarks))


# --- EXTRACTED CELL ---

PoseWorldLandmarks

# --- EXTRACTED CELL ---

PoseWorldLandmarks[0][0]

# --- EXTRACTED CELL ---


import numpy as np
PoseWorldLandmarksArray = np.array(PoseWorldLandmarks[0])


# --- EXTRACTED CELL ---

PoseWorldLandmarksArray.shape

# --- EXTRACTED CELL ---

PoseWorldLandmarksArray

# --- EXTRACTED CELL ---

PoseWorldLandmarksArray[0].x

# --- EXTRACTED CELL ---

RightHandWorldLandmarksArray = np.array(RightHandWorldLandmarks)
print(RightHandWorldLandmarksArray.shape)


# --- EXTRACTED CELL ---

LeftHandWorldLandmarksArray = np.array(LeftHandWorldLandmarks)
print(LeftHandWorldLandmarksArray.shape)


# --- EXTRACTED CELL ---

RightHandList = []
LeftHandList = [] 
print(len(RightHandWorldLandmarksArray))
for i in range(21):
    RightHandList.append(RightHandWorldLandmarksArray[i].x)
    RightHandList.append(RightHandWorldLandmarksArray[i].y)
    RightHandList.append(RightHandWorldLandmarksArray[i].z)
for i in range(21):
    LeftHandList.append(LeftHandWorldLandmarksArray[i].x)
    LeftHandList.append(LeftHandWorldLandmarksArray[i].y)
    LeftHandList.append(LeftHandWorldLandmarksArray[i].z)


# --- EXTRACTED CELL ---

PoseWorldLandmarksArray.shape

# --- EXTRACTED CELL ---

PoseWorldLandmarksArray.shape

# --- EXTRACTED CELL ---

PoseList = []
for i in range(33):
    PoseList.append(PoseWorldLandmarksArray[i].x)
    PoseList.append(PoseWorldLandmarksArray[i].y)
    PoseList.append(PoseWorldLandmarksArray[i].z)

# --- EXTRACTED CELL ---

PoseWorldLandmarks
print(len(RightHandList))
print(len(LeftHandList))
print(len(PoseList))


# --- EXTRACTED CELL ---

RightHandList = np.array(RightHandList)
LeftHandList = np.array(LeftHandList)
PoseList =  np.array(PoseList)

# --- EXTRACTED CELL ---

PoseList.shape
PoseList = PoseList.reshape(-1, 3)
RightHandList.shape
RightHandList = RightHandList.reshape(-1, 3)
LeftHandList.shape
LeftHandList = LeftHandList.reshape(-1, 3)

# --- EXTRACTED CELL ---

print(RightHandList.shape)
print(LeftHandList.shape)
print(PoseList.shape)

# --- EXTRACTED CELL ---

RightHandList[0]

# --- EXTRACTED CELL ---

LeftHandList[0]

# --- EXTRACTED CELL ---

import json

data = {
    "right_hand": RightHandList.tolist(),
    "left_hand": LeftHandList.tolist(),
    "pose": PoseList.tolist()
}

with open("Outputs/TasksLandmarksTpose.json", "w") as f:
    json.dump(data, f, indent=4)

# --- EXTRACTED CELL ---

RightShoulderGlobalRotations = []
LeftShoulderGlobalRotations = []
RightThumbBaseGlobalRotations = []
RightThumbMcpGlobalRotations = []
RightThumbIpGlobalRotations = []
RightThumbTipGlobalRotations = []
RightIndexBaseGlobalRotations = []
RightMiddleBaseGlobalRotations = []
RightRingBaseGlobalRotations = []
RightPinkyBaseGlobalRotations = []
RightIndexMcpGlobalRotations = []
RightIndexDipGlobalRotations = []
RightIndexTipGlobalRotations = []
RightMiddleMcpGlobalRotations = []
RightMiddleDipGlobalRotations = []
RightMiddleTipGlobalRotations = []
RightRingMcpGlobalRotations = []
RightRingDipGlobalRotations = []
RightRingTipGlobalRotations = []
RightPinkyMcpGlobalRotations = []
RightPinkyDipGlobalRotations = []
RightPinkyTipGlobalRotations = []


# --- EXTRACTED CELL ---

LeftThumbBaseGlobalRotations = []
LeftThumbMcpGlobalRotations = []
LeftThumbIpGlobalRotations = []
LeftThumbTipGlobalRotations = []
LeftIndexBaseGlobalRotations = []
LeftMiddleBaseGlobalRotations = []
LeftRingBaseGlobalRotations = []
LeftPinkyBaseGlobalRotations = []
LeftIndexMcpGlobalRotations = []
LeftIndexDipGlobalRotations = []
LeftIndexTipGlobalRotations = []
LeftMiddleMcpGlobalRotations = []
LeftMiddleDipGlobalRotations = []
LeftMiddleTipGlobalRotations = []
LeftRingMcpGlobalRotations = []
LeftRingDipGlobalRotations = []
LeftRingTipGlobalRotations = []
LeftPinkyMcpGlobalRotations = []
LeftPinkyDipGlobalRotations = []
LeftPinkyTipGlobalRotations = []

# --- EXTRACTED CELL ---

RightUpperArmGlobalRotations = []
RightForeArmGlobalRotations = []

# --- EXTRACTED CELL ---

LeftUpperArmGlobalRotations = []
LeftForeArmGlobalRotations = []

# --- EXTRACTED CELL ---

import json 
with open("Outputs/TasksLandmarksTpose.json","r") as f:
    taskLandmarksData = json.load(f)

# --- EXTRACTED CELL ---

pose = taskLandmarksData['pose']
right_hand = taskLandmarksData['right_hand']
left_hand = taskLandmarksData['left_hand']

# --- EXTRACTED CELL ---

import numpy as np
right_hand = np.array(right_hand)
left_hand = np.array(left_hand)
pose = np.array(pose)

# --- EXTRACTED CELL ---

pose.shape

# --- EXTRACTED CELL ---

pose[:,1] *= -1
pose[:,2] *= -1
right_hand[:,1] *= -1
right_hand[:,2] *= -1
left_hand[:,1] *= -1
left_hand[:,2] *= -1

# --- EXTRACTED CELL ---

imgShouldersMidpoint = (np.array(pose[11]) + np.array(pose[12])) / 2

# --- EXTRACTED CELL ---

imgRightShoulderVector = np.array(pose[12]) - imgShouldersMidpoint
imgLeftShoulderVector = np.array(pose[11]) - imgShouldersMidpoint
imgRightUpperArmVector = np.array(pose[14]) - np.array(pose[12])   
imgRightForeArmVector = np.array(pose[16]) - np.array(pose[14])
imgRightThumbBaseVector = np.array(right_hand[1]) - np.array(right_hand[0])
imgRightThumbMcpVector = np.array(right_hand[2]) - np.array(right_hand[1])
imgRightThumbIpVector = np.array(right_hand[3]) - np.array(right_hand[2])
imgRightThumbTipVector = np.array(right_hand[4]) - np.array(right_hand[3])
imgRightIndexBaseVector = np.array(right_hand[5]) - np.array(right_hand[0])
imgRightIndexMcpVector = np.array(right_hand[6]) - np.array(right_hand[5])
imgRightIndexDipVector = np.array(right_hand[7]) - np.array(right_hand[6])
imgRightIndexTipVector = np.array(right_hand[8]) - np.array(right_hand[7])
imgRightMiddleBaseVector = np.array(right_hand[9]) - np.array(right_hand[0])
imgRightMiddleMcpVector = np.array(right_hand[10]) - np.array(right_hand[9])
imgRightMiddleDipVector = np.array(right_hand[11]) - np.array(right_hand[10])
imgRightMiddleTipVector = np.array(right_hand[12]) - np.array(right_hand[11])
imgRightRingBaseVector = np.array(right_hand[13]) - np.array(right_hand[0])
imgRightRingMcpVector = np.array(right_hand[14]) - np.array(right_hand[13])
imgRightRingDipVector = np.array(right_hand[15]) - np.array(right_hand[14])
imgRightRingTipVector = np.array(right_hand[16]) - np.array(right_hand[15])
imgRightPinkyBaseVector = np.array(right_hand[17]) - np.array(right_hand[0])
imgRightPinkyMcpVector = np.array(right_hand[18]) - np.array(right_hand[17])
imgRightPinkyDipVector = np.array(right_hand[19]) - np.array(right_hand[18])
imgRightPinkyTipVector = np.array(right_hand[20]) - np.array(right_hand[19])
imgLeftUpperArmVector = np.array(pose[13]) - np.array(pose[11])
imgLeftForeArmVector = np.array(pose[15]) - np.array(pose[13])
imgLeftThumbBaseVector = np.array(left_hand[1]) - np.array(left_hand[0])
imgLeftThumbMcpVector = np.array(left_hand[2]) - np.array(left_hand[1])
imgLeftThumbIpVector = np.array(left_hand[3]) - np.array(left_hand[2])
imgLeftThumbTipVector = np.array(left_hand[4]) - np.array(left_hand[3])
imgLeftIndexBaseVector = np.array(left_hand[5]) - np.array(left_hand[0])
imgLeftIndexMcpVector = np.array(left_hand[6]) - np.array(left_hand[5])
imgLeftIndexDipVector = np.array(left_hand[7]) - np.array(left_hand[6])
imgLeftIndexTipVector = np.array(left_hand[8]) - np.array(left_hand[7])
imgLeftMiddleBaseVector = np.array(left_hand[9]) - np.array(left_hand[0])
imgLeftMiddleMcpVector = np.array(left_hand[10]) - np.array(left_hand[9])
imgLeftMiddleDipVector = np.array(left_hand[11]) - np.array(left_hand[10])
imgLeftMiddleTipVector = np.array(left_hand[12]) - np.array(left_hand[11])
imgLeftRingBaseVector = np.array(left_hand[13]) - np.array(left_hand[0])
imgLeftRingMcpVector = np.array(left_hand[14]) - np.array(left_hand[13])
imgLeftRingDipVector = np.array(left_hand[15]) - np.array(left_hand[14])
imgLeftRingTipVector = np.array(left_hand[16]) - np.array(left_hand[15])
imgLeftPinkyBaseVector = np.array(left_hand[17]) - np.array(left_hand[0])
imgLeftPinkyMcpVector = np.array(left_hand[18]) - np.array(left_hand[17])
imgLeftPinkyDipVector = np.array(left_hand[19]) - np.array(left_hand[18])
imgLeftPinkyTipVector = np.array(left_hand[20]) - np.array(left_hand[19])

# --- EXTRACTED CELL ---

import json
allFrameRightHandWorldLandmarks = []
allFrameLeftHandWorldLandmarks = []

# STEP 1: Import the necessary modules.
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.VIDEO,num_hands=2)
HandDetector = vision.HandLandmarker.create_from_options(options)


import cv2
cap = cv2.VideoCapture('Inputs/Apple.mp4')
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    VideoFrame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(
    cap.get(cv2.CAP_PROP_POS_MSEC)
    )
    detection_result = HandDetector.detect_for_video(VideoFrame,timestamp_ms)
    length = len(detection_result.handedness)
    if (length == 2):
        if(detection_result.handedness[0][0].category_name == "Right"):
            allFrameRightHandWorldLandmarks.append(detection_result.hand_world_landmarks[0])
        elif(detection_result.handedness[1][0].category_name == "Right"):
            allFrameRightHandWorldLandmarks.append(detection_result.hand_world_landmarks[1])
        else:
            allFrameRightHandWorldLandmarks.append(None)
        if(detection_result.handedness[1][0].category_name == "Left"):
            allFrameLeftHandWorldLandmarks.append(detection_result.hand_world_landmarks[1])
        elif(detection_result.handedness[0][0].category_name == "Left"):
            allFrameLeftHandWorldLandmarks.append(detection_result.hand_world_landmarks[0])
        else:
            allFrameLeftHandWorldLandmarks.append(None)
    elif (length == 1):
        if(detection_result.handedness[0][0].category_name == "Right"):
            allFrameRightHandWorldLandmarks.append(detection_result.hand_world_landmarks[0])
            allFrameLeftHandWorldLandmarks.append(None)
        elif(detection_result.handedness[0][0].category_name == "Left"):
            allFrameLeftHandWorldLandmarks.append(detection_result.hand_world_landmarks[0])
            allFrameRightHandWorldLandmarks.append(None)
    else:
        allFrameRightHandWorldLandmarks.append(None)
        allFrameLeftHandWorldLandmarks.append(None)
cap.release()


len(allFrameLeftHandWorldLandmarks)

len(allFrameRightHandWorldLandmarks)

# Make points like image skeleton
RightHandList = []
LeftHandList = []

PoseList = []



for i in range(len(allFrameRightHandWorldLandmarks)):
 for j in range(21):
    if allFrameRightHandWorldLandmarks[i] is not None:
        RightHandList.append(allFrameRightHandWorldLandmarks[i][j].x)
        RightHandList.append(allFrameRightHandWorldLandmarks[i][j].y)
        RightHandList.append(allFrameRightHandWorldLandmarks[i][j].z)
    elif allFrameRightHandWorldLandmarks[i] is None:
        RightHandList.append(0)
        RightHandList.append(0)
        RightHandList.append(0)

len(RightHandList)

allFramePoseWorldLandmarks = []

for i in range(len(allFrameLeftHandWorldLandmarks)):
 for j in range(21):
    if allFrameLeftHandWorldLandmarks[i] is not None:
        LeftHandList.append(allFrameLeftHandWorldLandmarks[i][j].x)
        LeftHandList.append(allFrameLeftHandWorldLandmarks[i][j].y)
        LeftHandList.append(allFrameLeftHandWorldLandmarks[i][j].z)
    elif allFrameLeftHandWorldLandmarks[i] is None:
        LeftHandList.append(0)
        LeftHandList.append(0)
        LeftHandList.append(0)


len(LeftHandList)

import numpy as np
LeftHandList  = np.array(LeftHandList)
LeftHandList = LeftHandList.reshape(85,21,3)


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,running_mode=vision.RunningMode.VIDEO
    )
PoseDetector = vision.PoseLandmarker.create_from_options(options)



allFramePoseWorldLandmarks = []

import cv2
# STEP 3: Load the input image.
cap = cv2.VideoCapture('Inputs/Apple.mp4')
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    VideoFrame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(
    cap.get(cv2.CAP_PROP_POS_MSEC)
    )
    detection_result = PoseDetector.detect_for_video(VideoFrame,timestamp_ms)
    allFramePoseWorldLandmarks.append(detection_result.pose_world_landmarks)
cap.release()
right_hand = np.array(right_hand)
import numpy as np
RightHandList  = np.array(RightHandList)
RightHandList = RightHandList.reshape(85,21,3)
PoseList = []
for i in range(len(allFramePoseWorldLandmarks)):
    for j in range(33):
        if(len(allFramePoseWorldLandmarks[i]) > 0):
            PoseList.append(allFramePoseWorldLandmarks[i][0][j].x)
            PoseList.append(allFramePoseWorldLandmarks[i][0][j].y)
            PoseList.append(allFramePoseWorldLandmarks[i][0][j].z)
        elif (len(allFramePoseWorldLandmarks[i]) == 0):
            PoseList.append(0)
            PoseList.append(0)
            PoseList.append(0)

            
PoseList = np.array(PoseList)
PoseList = PoseList.reshape(85,33,3)
len(PoseList)

print(PoseList.shape)
print(RightHandList.shape)
print(LeftHandList.shape)


RightHandList[:, :, 1] *= -1 
RightHandList[:, :, 2] *= -1 

LeftHandList[:, :, 1] *= -1 
LeftHandList[:, :, 2] *= -1 

PoseList[:, :, 1] *= -1 
PoseList[:, :, 2] *= -1

# --- EXTRACTED CELL ---

allFramesShouldersMidpoint = []
for i in range(85):
    allFramesShouldersMidpoint.append((PoseList[i][11] + PoseList[i][12]) / 2) 

# --- EXTRACTED CELL ---

allFrameRightShoulderVector = []
for i in range(85):
    allFrameRightShoulderVector.append(PoseList[i][12] - allFramesShouldersMidpoint[i])
allFrameLeftShoulderVector = []
for i in range(85):
    allFrameLeftShoulderVector.append(PoseList[i][11] - allFramesShouldersMidpoint[i])
allFrameRightUpperArmVector = []
for i in range(85):
    allFrameRightUpperArmVector.append(PoseList[i][14] - PoseList[i][12])
allFrameRightForeArmVector = []
for i in range(85):
    allFrameRightForeArmVector.append(PoseList[i][16] - PoseList[i][14])
allFrameRightThumbBaseVector = []
for i in range(85):
    allFrameRightThumbBaseVector.append(RightHandList[i][1] - RightHandList[i][0])
allFrameRightThumbMcpVector = []
for i in range(85):
    allFrameRightThumbMcpVector.append(RightHandList[i][2] - RightHandList[i][1])
allFrameRightThumbIpVector = []
for i in range(85):
    allFrameRightThumbIpVector.append(RightHandList[i][3] - RightHandList[i][2])
allFrameRightThumbTipVector = []
for i in range(85):
    allFrameRightThumbTipVector.append(RightHandList[i][4] - RightHandList[i][3])
allFrameRightIndexBaseVector = []
for i in range(85):
    allFrameRightIndexBaseVector.append(RightHandList[i][5] - RightHandList[i][0])
allFrameRightIndexMcpVector = []
for i in range(85):
    allFrameRightIndexMcpVector.append(RightHandList[i][6] - RightHandList[i][5])
allFrameRightIndexDipVector = []
for i in range(85):
    allFrameRightIndexDipVector.append(RightHandList[i][7] - RightHandList[i][6])
allFrameRightIndexTipVector = []
for i in range(85):
    allFrameRightIndexTipVector.append(RightHandList[i][8] - RightHandList[i][7])
allFrameRightMiddleBaseVector = []
for i in range(85):
    allFrameRightMiddleBaseVector.append(RightHandList[i][9] - RightHandList[i][0])
allFrameRightMiddleMcpVector = []
for i in range(85):
    allFrameRightMiddleMcpVector.append(RightHandList[i][10] - RightHandList[i][9])
allFrameRightMiddleDipVector = []
for i in range(85):
    allFrameRightMiddleDipVector.append(RightHandList[i][11] - RightHandList[i][10])
allFrameRightMiddleTipVector = []
for i in range(85):
    allFrameRightMiddleTipVector.append(RightHandList[i][12] - RightHandList[i][11])
allFrameRightRingBaseVector = []
for i in range(85):
    allFrameRightRingBaseVector.append(RightHandList[i][13] - RightHandList[i][0])
allFrameRightRingMcpVector = []
for i in range(85):
    allFrameRightRingMcpVector.append(RightHandList[i][14] - RightHandList[i][13])  
allFrameRightRingDipVector = []
for i in range(85):
    allFrameRightRingDipVector.append(RightHandList[i][15] - RightHandList[i][14])
allFrameRightRingTipVector = []
for i in range(85):
    allFrameRightRingTipVector.append(RightHandList[i][16] - RightHandList[i][15])
allFrameRightPinkyBaseVector = []
for i in range(85):
    allFrameRightPinkyBaseVector.append(RightHandList[i][17] - RightHandList[i][0])
allFrameRightPinkyMcpVector = []
for i in range(85):
    allFrameRightPinkyMcpVector.append(RightHandList[i][18] - RightHandList[i][17])
allFrameRightPinkyDipVector = []
for i in range(85):
    allFrameRightPinkyDipVector.append(RightHandList[i][19] - RightHandList[i][18])
allFrameRightPinkyTipVector = []
for i in range(85):
    allFrameRightPinkyTipVector.append(RightHandList[i][20] - RightHandList[i][19])


# --- EXTRACTED CELL ---

allFrameLeftUpperArmVector = []
for i in range(85):
    allFrameLeftUpperArmVector.append(PoseList[i][13] - PoseList[i][11])
allFrameLeftForeArmVector = []
for i in range(85):
    allFrameLeftForeArmVector.append(PoseList[i][15] - PoseList[i][13])
allFrameLeftThumbBaseVector = []
for i in range(85):
    allFrameLeftThumbBaseVector.append(LeftHandList[i][1] - LeftHandList[i][0])
allFrameLeftThumbMcpVector = []
for i in range(85):
    allFrameLeftThumbMcpVector.append(LeftHandList[i][2] - LeftHandList[i][1])
allFrameLeftThumbIpVector = []
for i in range(85):
    allFrameLeftThumbIpVector.append(LeftHandList[i][3] - LeftHandList[i][2])
allFrameLeftThumbTipVector = []
for i in range(85):
    allFrameLeftThumbTipVector.append(LeftHandList[i][4] - LeftHandList[i][3])
allFrameLeftIndexBaseVector = []
for i in range(85):
    allFrameLeftIndexBaseVector.append(LeftHandList[i][5] - LeftHandList[i][0])
allFrameLeftIndexMcpVector = []
for i in range(85):
    allFrameLeftIndexMcpVector.append(LeftHandList[i][6] - LeftHandList[i][5])
allFrameLeftIndexDipVector = []
for i in range(85):
    allFrameLeftIndexDipVector.append(LeftHandList[i][7] - LeftHandList[i][6])
allFrameLeftIndexTipVector = []
for i in range(85):
    allFrameLeftIndexTipVector.append(LeftHandList[i][8] - LeftHandList[i][7])
allFrameLeftMiddleBaseVector = []
for i in range(85):
    allFrameLeftMiddleBaseVector.append(LeftHandList[i][9] - LeftHandList[i][0])
allFrameLeftMiddleMcpVector = []
for i in range(85):
    allFrameLeftMiddleMcpVector.append(LeftHandList[i][10] - LeftHandList[i][9])
allFrameLeftMiddleDipVector = []
for i in range(85):
    allFrameLeftMiddleDipVector.append(LeftHandList[i][11] - LeftHandList[i][10])
allFrameLeftMiddleTipVector = []
for i in range(85):
    allFrameLeftMiddleTipVector.append(LeftHandList[i][12] - LeftHandList[i][11])
allFrameLeftRingBaseVector = []
for i in range(85):
    allFrameLeftRingBaseVector.append(LeftHandList[i][13] - LeftHandList[i][0])
allFrameLeftRingMcpVector = []
for i in range(85):
    allFrameLeftRingMcpVector.append(LeftHandList[i][14] - LeftHandList[i][13])
allFrameLeftRingDipVector = []
for i in range(85):
    allFrameLeftRingDipVector.append(LeftHandList[i][15] - LeftHandList[i][14])
allFrameLeftRingTipVector = []
for i in range(85):
    allFrameLeftRingTipVector.append(LeftHandList[i][16] - LeftHandList[i][15])
allFrameLeftPinkyBaseVector = []
for i in range(85):
    allFrameLeftPinkyBaseVector.append(LeftHandList[i][17] - LeftHandList[i][0])
allFrameLeftPinkyMcpVector = []
for i in range(85):
    allFrameLeftPinkyMcpVector.append(LeftHandList[i][18] - LeftHandList[i][17])
allFrameLeftPinkyDipVector = []
for i in range(85):
    allFrameLeftPinkyDipVector.append(LeftHandList[i][19] - LeftHandList[i][18])
allFrameLeftPinkyTipVector = []
for i in range(85):
    allFrameLeftPinkyTipVector.append(LeftHandList[i][20] - LeftHandList[i][19])

# --- EXTRACTED CELL ---

imgLeftPinkyCrossIndexBaseVector = np.cross(imgLeftIndexBaseVector, imgLeftPinkyBaseVector)

# --- EXTRACTED CELL ---

def build_bone_matrix_with_z_lock_ArmChain(bone_vec, z_normal):
    # 1. True Y-Axis
    if np.linalg.norm(bone_vec) == 0  or np.linalg.norm(z_normal) == 0:
        if np.linalg.norm(bone_vec) == 0:
            y = [0, 1, 0]
        if np.linalg.norm(z_normal) == 0:
            z = [0, 0, 1]
        x = [1, 0, 0]
    elif np.linalg.norm(bone_vec) != 0 and np.linalg.norm(z_normal) != 0:
        y = bone_vec / np.linalg.norm(bone_vec)
        z_temp = z_normal / np.linalg.norm(z_normal)
        x = np.cross(y,z_temp)
        x = x / np.linalg.norm(x)
        z = np.cross(x,y)    
    return np.column_stack((x, y, z)) # Perfectly orthogonal 3x3 Matrix!


def get_stable_normal(bone_vec, preferred_normal, fallback_normal):
    if preferred_normal is not None and np.linalg.norm(preferred_normal) > 1e-4:
        cross_prod = np.cross(bone_vec, preferred_normal)
        if np.linalg.norm(cross_prod) > 1e-4:
            return preferred_normal
            
    if fallback_normal is not None and np.linalg.norm(fallback_normal) > 1e-4:
        cross_prod = np.cross(bone_vec, fallback_normal)
        if np.linalg.norm(cross_prod) > 1e-4:
            return fallback_normal
            
    # Ultimate fallback: perpendicular vector
    norm_val = np.linalg.norm(bone_vec)
    if norm_val < 1e-5:
        return np.array([0.0, 0.0, 1.0])
    if abs(bone_vec[0]) > 0.9 * norm_val:
        return np.array([0.0, 1.0, 0.0])
    else:
        return np.array([1.0, 0.0, 0.0])

def nlerp_vector(v0, v1, t):
    v0_norm = v0 / np.linalg.norm(v0)
    v1_norm = v1 / np.linalg.norm(v1)
    if np.dot(v0_norm, v1_norm) < 0:
        v1_norm = -v1_norm
    res = (1.0 - t) * v0_norm + t * v1_norm
    return res / np.linalg.norm(res)

def get_forearm_normal(bone_vec, palm_normal, upper_arm_normal, chest_normal, t=0.5):
    p_norm = get_stable_normal(bone_vec, palm_normal, chest_normal)
    u_norm = get_stable_normal(bone_vec, upper_arm_normal, chest_normal)
    if p_norm is None or np.all(p_norm == 0) or np.linalg.norm(p_norm) < 1e-4:
        return u_norm
    return nlerp_vector(u_norm, p_norm, t)

def get_locked_global_rot_ArmChain(img_bone_vec, img_palm_normal, frame_bone_vec, frame_palm_normal, img_fallback_normal=None, frame_fallback_normal=None):
    if np.linalg.norm(frame_bone_vec) == 0:
        return np.eye(3)
        
    resolved_img_normal = get_stable_normal(img_bone_vec, img_palm_normal, img_fallback_normal)
    resolved_frame_normal = get_stable_normal(frame_bone_vec, frame_palm_normal, frame_fallback_normal)
    
    R_rest = build_bone_matrix_with_z_lock_ArmChain(img_bone_vec, resolved_img_normal)
    R_curr = build_bone_matrix_with_z_lock_ArmChain(frame_bone_vec, resolved_frame_normal)
    return R_curr @ np.transpose(R_rest)   

# --- EXTRACTED CELL ---

imgMidHips = (np.array(pose[23]) + np.array(pose[24])) / 2
imgSpine = imgMidHips-imgShouldersMidpoint

# --- EXTRACTED CELL ---

double = np.array(pose)

# --- EXTRACTED CELL ---

double.shape

# --- EXTRACTED CELL ---

imgShoulderSpanVector = pose[12] - pose[11]

# --- EXTRACTED CELL ---

imgChestNormal = np.cross(imgShoulderSpanVector, imgSpine)

# --- EXTRACTED CELL ---

allFrameHipMidPoints = []
for i in range(85):
    allFrameHipMidPoints.append((PoseList[i][23] + PoseList[i][24]) / 2)

# --- EXTRACTED CELL ---

allFrameSpineVector = []
for i in range(85):
    allFrameSpineVector.append(allFrameHipMidPoints[i]-allFramesShouldersMidpoint[i])

# --- EXTRACTED CELL ---

allFrameShoulderSpanVector = []
for i in range(85):
    allFrameShoulderSpanVector.append(PoseList[i][12]-PoseList[i][11])

# --- EXTRACTED CELL ---

allFrameChestNormal = []
for i in range(85):
    allFrameChestNormal.append(np.cross(allFrameShoulderSpanVector[i], allFrameSpineVector[i]))
allFrameRightUpperArmNormal = []
for i in range(85):
    allFrameRightUpperArmNormal.append(np.cross(allFrameRightForeArmVector[i], allFrameRightUpperArmVector[i]))
allFramePalmNormalRightHand = []
for i in range(85):
    allFramePalmNormalRightHand.append(np.cross(allFrameRightPinkyBaseVector[i], allFrameRightIndexBaseVector[i]))
imgRightUpperArmNormal = np.cross(imgRightForeArmVector, imgRightUpperArmVector)
imgPalmNormalRightHand = np.cross(imgRightPinkyBaseVector, imgRightIndexBaseVector)
    
allFrameLeftUpperArmNormal = []
for i in range(85):
    allFrameLeftUpperArmNormal.append(np.cross(allFrameLeftForeArmVector[i], allFrameLeftUpperArmVector[i]))
allFramePalmNormalLeftHand = []
for i in range(85):
    allFramePalmNormalLeftHand.append(np.cross(allFrameLeftIndexBaseVector[i], allFrameLeftPinkyBaseVector[i]))
imgLeftUpperArmNormal = np.cross(imgLeftForeArmVector, imgLeftUpperArmVector)
imgPalmNormalLeftHand = np.cross(imgLeftIndexBaseVector, imgLeftPinkyBaseVector)

# --- EXTRACTED CELL ---

                 

# Overwrite the old align_vectors with our new perfect math!
for i in range(85): # Change to 132 for your Red.mp4 video!
    RightShoulderGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightShoulderVector,  imgChestNormal, allFrameRightShoulderVector[i], allFrameChestNormal[i]))
    LeftShoulderGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftShoulderVector, imgChestNormal, allFrameLeftShoulderVector[i], allFrameChestNormal[i]))
    RightUpperArmGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightUpperArmVector, imgRightUpperArmNormal, allFrameRightUpperArmVector[i], allFrameRightUpperArmNormal[i], imgChestNormal, allFrameChestNormal[i]))
    img_right_forearm_normal = get_forearm_normal(imgRightForeArmVector, imgPalmNormalRightHand, imgRightUpperArmNormal, imgChestNormal)
    frame_right_forearm_normal = get_forearm_normal(allFrameRightForeArmVector[i], allFramePalmNormalRightHand[i], allFrameRightUpperArmNormal[i], allFrameChestNormal[i])
    RightForeArmGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightForeArmVector, img_right_forearm_normal, allFrameRightForeArmVector[i], frame_right_forearm_normal))
    
    RightThumbBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightThumbBaseVector, imgPalmNormalRightHand, allFrameRightThumbBaseVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightThumbMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightThumbMcpVector, imgPalmNormalRightHand, allFrameRightThumbMcpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightThumbIpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightThumbIpVector, imgPalmNormalRightHand, allFrameRightThumbIpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightThumbTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightThumbTipVector, imgPalmNormalRightHand, allFrameRightThumbTipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    
    
    # --- RIGHT HAND ---
    RightIndexBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightIndexBaseVector, imgPalmNormalRightHand, allFrameRightIndexBaseVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightIndexMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightIndexMcpVector, imgPalmNormalRightHand, allFrameRightIndexMcpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightIndexDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightIndexDipVector, imgPalmNormalRightHand, allFrameRightIndexDipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightIndexTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightIndexTipVector, imgPalmNormalRightHand, allFrameRightIndexTipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    
    RightMiddleBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightMiddleBaseVector, imgPalmNormalRightHand, allFrameRightMiddleBaseVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightMiddleMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightMiddleMcpVector, imgPalmNormalRightHand, allFrameRightMiddleMcpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightMiddleDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightMiddleDipVector, imgPalmNormalRightHand, allFrameRightMiddleDipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightMiddleTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightMiddleTipVector, imgPalmNormalRightHand, allFrameRightMiddleTipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    RightRingBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightRingBaseVector, imgPalmNormalRightHand, allFrameRightRingBaseVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightRingMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightRingMcpVector, imgPalmNormalRightHand, allFrameRightRingMcpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightRingDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightRingDipVector, imgPalmNormalRightHand, allFrameRightRingDipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightRingTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightRingTipVector, imgPalmNormalRightHand, allFrameRightRingTipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    
    RightPinkyBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightPinkyBaseVector, imgPalmNormalRightHand, allFrameRightPinkyBaseVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightPinkyMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightPinkyMcpVector, imgPalmNormalRightHand, allFrameRightPinkyMcpVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightPinkyDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightPinkyDipVector, imgPalmNormalRightHand, allFrameRightPinkyDipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))
    RightPinkyTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgRightPinkyTipVector, imgPalmNormalRightHand, allFrameRightPinkyTipVector[i], allFramePalmNormalRightHand[i], imgChestNormal, allFrameChestNormal[i]))

    # --- LEFT HAND ---
    
    LeftUpperArmGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftUpperArmVector, imgLeftUpperArmNormal, allFrameLeftUpperArmVector[i], allFrameLeftUpperArmNormal[i], imgChestNormal, allFrameChestNormal[i]))
    img_left_forearm_normal = get_forearm_normal(imgLeftForeArmVector, imgPalmNormalLeftHand, imgLeftUpperArmNormal, imgChestNormal)
    frame_left_forearm_normal = get_forearm_normal(allFrameLeftForeArmVector[i], allFramePalmNormalLeftHand[i], allFrameLeftUpperArmNormal[i], allFrameChestNormal[i])
    LeftForeArmGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftForeArmVector, img_left_forearm_normal, allFrameLeftForeArmVector[i], frame_left_forearm_normal))
    
    
    LeftThumbBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftThumbBaseVector, imgPalmNormalLeftHand, allFrameLeftThumbBaseVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftThumbMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftThumbMcpVector, imgPalmNormalLeftHand, allFrameLeftThumbMcpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftThumbIpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftThumbIpVector, imgPalmNormalLeftHand, allFrameLeftThumbIpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftThumbTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftThumbTipVector, imgPalmNormalLeftHand, allFrameLeftThumbTipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    
    LeftIndexBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftIndexBaseVector, imgPalmNormalLeftHand, allFrameLeftIndexBaseVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftIndexMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftIndexMcpVector, imgPalmNormalLeftHand, allFrameLeftIndexMcpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftIndexDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftIndexDipVector, imgPalmNormalLeftHand, allFrameLeftIndexDipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftIndexTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftIndexTipVector, imgPalmNormalLeftHand, allFrameLeftIndexTipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    LeftMiddleBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftMiddleBaseVector, imgPalmNormalLeftHand, allFrameLeftMiddleBaseVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftMiddleMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftMiddleMcpVector, imgPalmNormalLeftHand, allFrameLeftMiddleMcpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftMiddleDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftMiddleDipVector, imgPalmNormalLeftHand, allFrameLeftMiddleDipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftMiddleTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftMiddleTipVector, imgPalmNormalLeftHand, allFrameLeftMiddleTipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    LeftRingBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftRingBaseVector, imgPalmNormalLeftHand, allFrameLeftRingBaseVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftRingMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftRingMcpVector, imgPalmNormalLeftHand, allFrameLeftRingMcpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftRingDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftRingDipVector, imgPalmNormalLeftHand, allFrameLeftRingDipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftRingTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftRingTipVector, imgPalmNormalLeftHand, allFrameLeftRingTipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    
    LeftPinkyBaseGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftPinkyBaseVector, imgPalmNormalLeftHand, allFrameLeftPinkyBaseVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftPinkyMcpGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftPinkyMcpVector, imgPalmNormalLeftHand, allFrameLeftPinkyMcpVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftPinkyDipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftPinkyDipVector, imgPalmNormalLeftHand, allFrameLeftPinkyDipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))
    LeftPinkyTipGlobalRotations.append(get_locked_global_rot_ArmChain(imgLeftPinkyTipVector, imgPalmNormalLeftHand, allFrameLeftPinkyTipVector[i], allFramePalmNormalLeftHand[i], imgChestNormal, allFrameChestNormal[i]))

# --- EXTRACTED CELL ---

def getLocalRotations(parent_global,child_global):
    return np.transpose(parent_global) @ child_global
allFrameRightUpperArmLocalRotations = []
allFrameLeftUpperArmLocalRotations = []
allFrameRightForeArmLocalRotations = []
allFrameRightThumbBaseLocalRotations = []
allFrameRightThumbMcpLocalRotations = []
allFrameRightThumbIpLocalRotations = []
allFrameRightThumbTipLocalRotations = []
allFrameRightIndexBaseLocalRotations = []
allFrameRightIndexMcpLocalRotations = []
allFrameRightIndexDipLocalRotations = []
allFrameRightIndexTipLocalRotations = []
allFrameRightMiddleBaseLocalRotations = []
allFrameRightMiddleMcpLocalRotations = []
allFrameRightMiddleDipLocalRotations = []
allFrameRightMiddleTipLocalRotations = []
allFrameRightRingBaseLocalRotations = []
allFrameRightRingMcpLocalRotations = []
allFrameRightRingDipLocalRotations = []
allFrameRightRingTipLocalRotations = []
allFrameRightPinkyBaseLocalRotations = []
allFrameRightPinkyMcpLocalRotations = []
allFrameRightPinkyDipLocalRotations = []
allFrameRightPinkyTipLocalRotations = []

allFrameLeftForeArmLocalRotations = []
allFrameLeftThumbBaseLocalRotations = []
allFrameLeftThumbMcpLocalRotations = []
allFrameLeftThumbIpLocalRotations = []
allFrameLeftThumbTipLocalRotations = []
allFrameLeftIndexBaseLocalRotations = []
allFrameLeftIndexMcpLocalRotations = []
allFrameLeftIndexDipLocalRotations = []
allFrameLeftIndexTipLocalRotations = []
allFrameLeftMiddleBaseLocalRotations = []
allFrameLeftMiddleMcpLocalRotations = []
allFrameLeftMiddleDipLocalRotations = []
allFrameLeftMiddleTipLocalRotations = []
allFrameLeftRingBaseLocalRotations = []
allFrameLeftRingMcpLocalRotations = []
allFrameLeftRingDipLocalRotations = []
allFrameLeftRingTipLocalRotations = []
allFrameLeftPinkyBaseLocalRotations = []
allFrameLeftPinkyMcpLocalRotations = []
allFrameLeftPinkyDipLocalRotations = []
allFrameLeftPinkyTipLocalRotations = []
for i in range(85):
    allFrameRightUpperArmLocalRotations.append(getLocalRotations(RightShoulderGlobalRotations[i],RightUpperArmGlobalRotations[i]))
    allFrameLeftUpperArmLocalRotations.append(getLocalRotations(LeftShoulderGlobalRotations[i],LeftUpperArmGlobalRotations[i]))
    allFrameRightForeArmLocalRotations.append(getLocalRotations(RightUpperArmGlobalRotations[i],RightForeArmGlobalRotations[i]))
    allFrameRightThumbBaseLocalRotations.append(getLocalRotations(RightForeArmGlobalRotations[i],RightThumbBaseGlobalRotations[i]))
    allFrameRightThumbMcpLocalRotations.append(getLocalRotations(RightThumbBaseGlobalRotations[i],RightThumbMcpGlobalRotations[i]))
    allFrameRightThumbIpLocalRotations.append(getLocalRotations(RightThumbMcpGlobalRotations[i],RightThumbIpGlobalRotations[i]))
    allFrameRightThumbTipLocalRotations.append(getLocalRotations(RightThumbIpGlobalRotations[i],RightThumbTipGlobalRotations[i]))
    
    allFrameRightIndexBaseLocalRotations.append(getLocalRotations(RightForeArmGlobalRotations[i],RightIndexBaseGlobalRotations[i]))
    allFrameRightIndexMcpLocalRotations.append(getLocalRotations(RightIndexBaseGlobalRotations[i],RightIndexMcpGlobalRotations[i]))
    allFrameRightIndexDipLocalRotations.append(getLocalRotations(RightIndexMcpGlobalRotations[i],RightIndexDipGlobalRotations[i]))
    allFrameRightIndexTipLocalRotations.append(getLocalRotations(RightIndexDipGlobalRotations[i],RightIndexTipGlobalRotations[i]))
    
    allFrameRightMiddleBaseLocalRotations.append(getLocalRotations(RightForeArmGlobalRotations[i],RightMiddleBaseGlobalRotations[i]))
    allFrameRightMiddleMcpLocalRotations.append(getLocalRotations(RightMiddleBaseGlobalRotations[i],RightMiddleMcpGlobalRotations[i]))
    allFrameRightMiddleDipLocalRotations.append(getLocalRotations(RightMiddleMcpGlobalRotations[i],RightMiddleDipGlobalRotations[i]))
    allFrameRightMiddleTipLocalRotations.append(getLocalRotations(RightMiddleDipGlobalRotations[i],RightMiddleTipGlobalRotations[i]))
    
    allFrameRightRingBaseLocalRotations.append(getLocalRotations(RightForeArmGlobalRotations[i],RightRingBaseGlobalRotations[i]))
    allFrameRightRingMcpLocalRotations.append(getLocalRotations(RightRingBaseGlobalRotations[i],RightRingMcpGlobalRotations[i]))
    allFrameRightRingDipLocalRotations.append(getLocalRotations(RightRingMcpGlobalRotations[i],RightRingDipGlobalRotations[i]))
    allFrameRightRingTipLocalRotations.append(getLocalRotations(RightRingDipGlobalRotations[i],RightRingTipGlobalRotations[i]))
    
    allFrameRightPinkyBaseLocalRotations.append(getLocalRotations(RightForeArmGlobalRotations[i],RightPinkyBaseGlobalRotations[i]))
    allFrameRightPinkyMcpLocalRotations.append(getLocalRotations(RightPinkyBaseGlobalRotations[i],RightPinkyMcpGlobalRotations[i]))
    allFrameRightPinkyDipLocalRotations.append(getLocalRotations(RightPinkyMcpGlobalRotations[i],RightPinkyDipGlobalRotations[i]))
    allFrameRightPinkyTipLocalRotations.append(getLocalRotations(RightPinkyDipGlobalRotations[i],RightPinkyTipGlobalRotations[i]))
    
    allFrameLeftForeArmLocalRotations.append(getLocalRotations(LeftUpperArmGlobalRotations[i],LeftForeArmGlobalRotations[i]))
    allFrameLeftThumbBaseLocalRotations.append(getLocalRotations(LeftForeArmGlobalRotations[i],LeftThumbBaseGlobalRotations[i]))
    allFrameLeftThumbMcpLocalRotations.append(getLocalRotations(LeftThumbBaseGlobalRotations[i],LeftThumbMcpGlobalRotations[i]))
    allFrameLeftThumbIpLocalRotations.append(getLocalRotations(LeftThumbMcpGlobalRotations[i],LeftThumbIpGlobalRotations[i]))
    allFrameLeftThumbTipLocalRotations.append(getLocalRotations(LeftThumbIpGlobalRotations[i],LeftThumbTipGlobalRotations[i]))
    allFrameLeftIndexBaseLocalRotations.append(getLocalRotations(LeftForeArmGlobalRotations[i],LeftIndexBaseGlobalRotations[i]))
    allFrameLeftIndexMcpLocalRotations.append(getLocalRotations(LeftIndexBaseGlobalRotations[i],LeftIndexMcpGlobalRotations[i]))
    allFrameLeftIndexDipLocalRotations.append(getLocalRotations(LeftIndexMcpGlobalRotations[i],LeftIndexDipGlobalRotations[i]))
    allFrameLeftIndexTipLocalRotations.append(getLocalRotations(LeftIndexDipGlobalRotations[i],LeftIndexTipGlobalRotations[i]))
    allFrameLeftMiddleBaseLocalRotations.append(getLocalRotations(LeftForeArmGlobalRotations[i],LeftMiddleBaseGlobalRotations[i]))
    allFrameLeftMiddleMcpLocalRotations.append(getLocalRotations(LeftMiddleBaseGlobalRotations[i],LeftMiddleMcpGlobalRotations[i]))
    allFrameLeftMiddleDipLocalRotations.append(getLocalRotations(LeftMiddleMcpGlobalRotations[i],LeftMiddleDipGlobalRotations[i]))
    allFrameLeftMiddleTipLocalRotations.append(getLocalRotations(LeftMiddleDipGlobalRotations[i],LeftMiddleTipGlobalRotations[i]))
    allFrameLeftRingBaseLocalRotations.append(getLocalRotations(LeftForeArmGlobalRotations[i],LeftRingBaseGlobalRotations[i]))
    allFrameLeftRingMcpLocalRotations.append(getLocalRotations(LeftRingBaseGlobalRotations[i],LeftRingMcpGlobalRotations[i]))
    allFrameLeftRingDipLocalRotations.append(getLocalRotations(LeftRingMcpGlobalRotations[i],LeftRingDipGlobalRotations[i]))
    allFrameLeftRingTipLocalRotations.append(getLocalRotations(LeftRingDipGlobalRotations[i],LeftRingTipGlobalRotations[i]))
    allFrameLeftPinkyBaseLocalRotations.append(getLocalRotations(LeftForeArmGlobalRotations[i],LeftPinkyBaseGlobalRotations[i]))
    allFrameLeftPinkyMcpLocalRotations.append(getLocalRotations(LeftPinkyBaseGlobalRotations[i],LeftPinkyMcpGlobalRotations[i]))
    allFrameLeftPinkyDipLocalRotations.append(getLocalRotations(LeftPinkyMcpGlobalRotations[i],LeftPinkyDipGlobalRotations[i]))
    allFrameLeftPinkyTipLocalRotations.append(getLocalRotations(LeftPinkyDipGlobalRotations[i],LeftPinkyTipGlobalRotations[i]))
    
    

# --- EXTRACTED CELL ---

import json
data = {
    "RightShoulderGlobalRotations": RightShoulderGlobalRotations,
    "RightUpperArmLocalRotations": allFrameRightUpperArmLocalRotations,
    "RightForeArmLocalRotations" : allFrameRightForeArmLocalRotations,
    "RightThumbBaseLocalRotations" : allFrameRightThumbBaseLocalRotations,
    "RightThumbMcpLocalRotations" : allFrameRightThumbMcpLocalRotations,
    "RightThumbIpLocalRotations" : allFrameRightThumbIpLocalRotations,
    "RightThumbTipLocalRotations" : allFrameRightThumbTipLocalRotations,
    "RightIndexBaseLocalRotations" : allFrameRightIndexBaseLocalRotations,
    "RightIndexMcpLocalRotations" : allFrameRightIndexMcpLocalRotations,
    "RightIndexDipLocalRotations" : allFrameRightIndexDipLocalRotations,
    "RightIndexTipLocalRotations" : allFrameRightIndexTipLocalRotations,
    "RightMiddleBaseLocalRotations" : allFrameRightMiddleBaseLocalRotations,
    "RightMiddleMcpLocalRotations" : allFrameRightMiddleMcpLocalRotations,
    "RightMiddleDipLocalRotations" : allFrameRightMiddleDipLocalRotations,
    "RightMiddleTipLocalRotations" : allFrameRightMiddleTipLocalRotations,
    "RightRingBaseLocalRotations" : allFrameRightRingBaseLocalRotations,
    "RightRingMcpLocalRotations" : allFrameRightRingMcpLocalRotations,
    "RightRingDipLocalRotations" : allFrameRightRingDipLocalRotations,
    "RightRingTipLocalRotations" : allFrameRightRingTipLocalRotations,
    "RightPinkyBaseLocalRotations" : allFrameRightPinkyBaseLocalRotations,
    "RightPinkyMcpLocalRotations" : allFrameRightPinkyMcpLocalRotations,
    "RightPinkyDipLocalRotations" : allFrameRightPinkyDipLocalRotations,
    "RightPinkyTipLocalRotations" : allFrameRightPinkyTipLocalRotations,
    "LeftShoulderGlobalRotations": LeftShoulderGlobalRotations,
    "LeftUpperArmLocalRotations": allFrameLeftUpperArmLocalRotations,
    "LeftForeArmLocalRotations" : allFrameLeftForeArmLocalRotations,
    "LeftThumbBaseLocalRotations" : allFrameLeftThumbBaseLocalRotations,
    "LeftThumbMcpLocalRotations" : allFrameLeftThumbMcpLocalRotations,
    "LeftThumbIpLocalRotations" : allFrameLeftThumbIpLocalRotations,
    "LeftThumbTipLocalRotations" : allFrameLeftThumbTipLocalRotations,
    "LeftIndexBaseLocalRotations" : allFrameLeftIndexBaseLocalRotations,
    "LeftIndexMcpLocalRotations" : allFrameLeftIndexMcpLocalRotations,
    "LeftIndexDipLocalRotations" : allFrameLeftIndexDipLocalRotations,
    "LeftIndexTipLocalRotations" : allFrameLeftIndexTipLocalRotations,
    "LeftMiddleBaseLocalRotations" : allFrameLeftMiddleBaseLocalRotations,
    "LeftMiddleMcpLocalRotations" : allFrameLeftMiddleMcpLocalRotations,
    "LeftMiddleDipLocalRotations" : allFrameLeftMiddleDipLocalRotations,
    "LeftMiddleTipLocalRotations" : allFrameLeftMiddleTipLocalRotations,
    "LeftRingBaseLocalRotations" : allFrameLeftRingBaseLocalRotations,
    "LeftRingMcpLocalRotations" : allFrameLeftRingMcpLocalRotations,
    "LeftRingDipLocalRotations" : allFrameLeftRingDipLocalRotations,
    "LeftRingTipLocalRotations" : allFrameLeftRingTipLocalRotations,
    "LeftPinkyBaseLocalRotations" : allFrameLeftPinkyBaseLocalRotations,
    "LeftPinkyMcpLocalRotations" : allFrameLeftPinkyMcpLocalRotations,
    "LeftPinkyDipLocalRotations" : allFrameLeftPinkyDipLocalRotations,
    "LeftPinkyTipLocalRotations" : allFrameLeftPinkyTipLocalRotations
}
with open('iteration-9-apple.json', 'w') as f:
    json.dump(data, f, indent=4, default=lambda x: x.tolist())

# --- EXTRACTED CELL ---

from scipy.spatial.transform import Rotation as R
import numpy as np
import json

# Helper function to convert a list of 3x3 matrices to [x, y, z, w] quaternions
def convert_matrices_to_quaternions(matrix_list):
    quat_list = []
    for mat in matrix_list:
        # Check if the matrix is valid (not empty or full of zeros)
        if mat is None or np.all(mat == 0):
            quat_list.append([0.0, 0.0, 0.0, 1.0]) # Identity Quaternion
        else:
            try:
                # scipy handles the matrix to quaternion conversion
                rot = R.from_matrix(mat)
                quat_list.append(rot.as_quat().tolist()) # Returns [x, y, z, w]
            except ValueError:
                # If a matrix got corrupted, default to rest pose
                quat_list.append([0.0, 0.0, 0.0, 1.0])
    return quat_list

# Build the final dictionary using your exact variable names, wrapped in the converter
data = {
    "RightShoulderGlobalRotations": convert_matrices_to_quaternions(RightShoulderGlobalRotations),
    "RightUpperArmLocalRotations": convert_matrices_to_quaternions(allFrameRightUpperArmLocalRotations),
    "RightForeArmLocalRotations" : convert_matrices_to_quaternions(allFrameRightForeArmLocalRotations),
    "RightThumbBaseLocalRotations" : convert_matrices_to_quaternions(allFrameRightThumbBaseLocalRotations),
    "RightThumbMcpLocalRotations" : convert_matrices_to_quaternions(allFrameRightThumbMcpLocalRotations),
    "RightThumbIpLocalRotations" : convert_matrices_to_quaternions(allFrameRightThumbIpLocalRotations),
    "RightThumbTipLocalRotations" : convert_matrices_to_quaternions(allFrameRightThumbTipLocalRotations),
    "RightIndexBaseLocalRotations" : convert_matrices_to_quaternions(allFrameRightIndexBaseLocalRotations),
    "RightIndexMcpLocalRotations" : convert_matrices_to_quaternions(allFrameRightIndexMcpLocalRotations),
    "RightIndexDipLocalRotations" : convert_matrices_to_quaternions(allFrameRightIndexDipLocalRotations),
    "RightIndexTipLocalRotations" : convert_matrices_to_quaternions(allFrameRightIndexTipLocalRotations),
    "RightMiddleBaseLocalRotations" : convert_matrices_to_quaternions(allFrameRightMiddleBaseLocalRotations),
    "RightMiddleMcpLocalRotations" : convert_matrices_to_quaternions(allFrameRightMiddleMcpLocalRotations),
    "RightMiddleDipLocalRotations" : convert_matrices_to_quaternions(allFrameRightMiddleDipLocalRotations),
    "RightMiddleTipLocalRotations" : convert_matrices_to_quaternions(allFrameRightMiddleTipLocalRotations),
    "RightRingBaseLocalRotations" : convert_matrices_to_quaternions(allFrameRightRingBaseLocalRotations),
    "RightRingMcpLocalRotations" : convert_matrices_to_quaternions(allFrameRightRingMcpLocalRotations),
    "RightRingDipLocalRotations" : convert_matrices_to_quaternions(allFrameRightRingDipLocalRotations),
    "RightRingTipLocalRotations" : convert_matrices_to_quaternions(allFrameRightRingTipLocalRotations),
    "RightPinkyBaseLocalRotations" : convert_matrices_to_quaternions(allFrameRightPinkyBaseLocalRotations),
    "RightPinkyMcpLocalRotations" : convert_matrices_to_quaternions(allFrameRightPinkyMcpLocalRotations),
    "RightPinkyDipLocalRotations" : convert_matrices_to_quaternions(allFrameRightPinkyDipLocalRotations),
    "RightPinkyTipLocalRotations" : convert_matrices_to_quaternions(allFrameRightPinkyTipLocalRotations),
    
    "LeftShoulderGlobalRotations": convert_matrices_to_quaternions(LeftShoulderGlobalRotations),
    "LeftUpperArmLocalRotations": convert_matrices_to_quaternions(allFrameLeftUpperArmLocalRotations),
    "LeftForeArmLocalRotations" : convert_matrices_to_quaternions(allFrameLeftForeArmLocalRotations),
    "LeftThumbBaseLocalRotations" : convert_matrices_to_quaternions(allFrameLeftThumbBaseLocalRotations),
    "LeftThumbMcpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftThumbMcpLocalRotations),
    "LeftThumbIpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftThumbIpLocalRotations),
    "LeftThumbTipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftThumbTipLocalRotations),
    "LeftIndexBaseLocalRotations" : convert_matrices_to_quaternions(allFrameLeftIndexBaseLocalRotations),
    "LeftIndexMcpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftIndexMcpLocalRotations),
    "LeftIndexDipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftIndexDipLocalRotations),
    "LeftIndexTipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftIndexTipLocalRotations),
    "LeftMiddleBaseLocalRotations" : convert_matrices_to_quaternions(allFrameLeftMiddleBaseLocalRotations),
    "LeftMiddleMcpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftMiddleMcpLocalRotations),
    "LeftMiddleDipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftMiddleDipLocalRotations),
    "LeftMiddleTipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftMiddleTipLocalRotations),
    "LeftRingBaseLocalRotations" : convert_matrices_to_quaternions(allFrameLeftRingBaseLocalRotations),
    "LeftRingMcpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftRingMcpLocalRotations),
    "LeftRingDipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftRingDipLocalRotations),
    "LeftRingTipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftRingTipLocalRotations),
    "LeftPinkyBaseLocalRotations" : convert_matrices_to_quaternions(allFrameLeftPinkyBaseLocalRotations),
    "LeftPinkyMcpLocalRotations" : convert_matrices_to_quaternions(allFrameLeftPinkyMcpLocalRotations),
    "LeftPinkyDipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftPinkyDipLocalRotations),
    "LeftPinkyTipLocalRotations" : convert_matrices_to_quaternions(allFrameLeftPinkyTipLocalRotations)
}

with open('iteration-9-apple-quats.json', 'w') as f:
    json.dump(data, f, indent=4)
print("Saved Quaternions successfully!")
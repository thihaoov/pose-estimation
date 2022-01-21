import cv2
import mediapipe as mp
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture("AITrainer/biceps2.mp4")
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    # img = cv2.imread("AITrainer/angle.jpg")

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if lmList:
        # Left Arm
        angle = detector.fingAngle(img, 11, 13, 15, True)
        # Right Arm
        angle = detector.fingAngle(img, 12, 14, 16, True)

        per = 100 - np.interp(angle, (40,150), (0,100))
        # print(int(angle), int(per))
        bar = np.interp(angle, (40,150), (150,700))

        # Check for dumbell curls numbers
        barColor = (255, 10, 100)
        if per == 100:
            barColor = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw curlc count
        cv2.rectangle(img, (50,940), (130,1005), (255,255,255), cv2.FILLED)
        cv2.putText(img, f'{str(int(count))}', (80,1000), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

        # Draw Bar
        cv2.rectangle(img, (50, 150), (125, 700), barColor, 2)
        cv2.rectangle(img, (50, int(bar)), (125,700), barColor, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, barColor, 4)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS:{str(int(fps))}', (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 4)

    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(30)
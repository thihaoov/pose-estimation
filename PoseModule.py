import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode= False, modelComplex=1, smooth = True, segment = False, 
                smoothSeg = True, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.modelComplex = modelComplex
        self.smooth = smooth
        self.segment = segment
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplex, self.smooth,
            self.segment, self.smoothSeg, self.detectionCon, self.trackingCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                            self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
        return self.lmList

    def fingAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle
        # print(int(angle))

        # Draw
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
            cv2.line(img, (x2,y2), (x3,y3), (255,0,0), 3)

            cv2.circle(img, (x1, y1), 10, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 10, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 10, (0,0,255), 2)

            # cv2.putText(img, str(int(angle)), (x2-20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        return angle


def main():
    cap = cv2.VideoCapture('PoseVideos/7.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[20])
            cv2.circle(img, (lmList[20][1],lmList[20][2]), 15, (255,0,0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{str(int(fps))}', (70,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0), 3)
        cv2.resize(img, (0,0), fx=0.7, fy=0.7)
        cv2.imshow("Image", img)
        cv2.waitKey(10)



if __name__ == "__main__":
    main()
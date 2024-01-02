import cv2 as cv
import mediapipe as mp
import time
import PoseModule as p

cap = cv.VideoCapture('PoseVideos/2.mp4')
pTime = 0
detector = p.poseDetector()

while True:
    sucess, img = cap.read()
    frame_resized = detector.rescaleFrame(img)
    frame_resized = detector.findPose(frame_resized,draw=False)
    imList = detector.findPosition(frame_resized,draw=False)
    if len(imList) != 0:
        print(imList[2])
        cv.circle(frame_resized,(imList[2][1],imList[2][2]),15,(0,0,255),cv.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(frame_resized, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("Video", frame_resized)
    cv.waitKey(1)
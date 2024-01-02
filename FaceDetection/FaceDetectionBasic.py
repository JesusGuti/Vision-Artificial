import cv2 as cv
import mediapipe as mp
import time

#Con webcam, pero hay un limite de Frames
cap = cv.VideoCapture(0)
pTime = 0
#Modulo para deteccion facial
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id,detection)
            #Este metodo nos dira que porcentaje tiene de ser una cara
            #print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img,bbox,(255,0,255),2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}', (20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv.imshow("Video",img)
    cv.waitKey(1)

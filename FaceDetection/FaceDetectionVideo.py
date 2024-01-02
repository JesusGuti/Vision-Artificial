import cv2 as cv
import mediapipe as mp
import time

#Con webcam, pero hay un limite de Frames
cap = cv.VideoCapture("FaceVideos/caras.mp4")
pTime = 0
#Modulo para deteccion facial
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection()

#Metodo para reescalar video
def rescaleFrame(frame,scale = 0.3):
    width = int(frame.shape[1]*0.3)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation = cv.INTER_AREA)


while True:
    success, img = cap.read()
    frame_resized = rescaleFrame(img)
    imgRGB = cv.cvtColor(frame_resized,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(id,detection)
            # Este metodo nos dira que porcentaje tiene de ser una cara
            #print(detection.score)
            #Nos dara la cara y sus datos especificos
            #print(detection.location_data.relative_bounding_box)
            #mpDraw.draw_detection(frame_resized,detection)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame_resized.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(frame_resized, bbox, (255, 0, 255), 2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(frame_resized,f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv.imshow("Video",frame_resized)
    cv.waitKey(1)

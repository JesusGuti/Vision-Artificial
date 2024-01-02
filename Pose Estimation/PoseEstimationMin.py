import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
#Obtenemos el video
#cap = cv.VideoCapture('PoseVideos/1.mp4')
cap = cv.VideoCapture(0)
#Metodo para reescalar video
def rescaleFrame(frame,scale = 0.30):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation = cv.INTER_AREA)


#
pTime = 0
while True:
    sucess, img = cap.read()
    #frame_resized = rescaleFrame(img)
    #Debemos convertir la imagen a RGB
    #imgRGB = cv.cvtColor(frame_resized,cv.COLOR_BGR2RGB)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #Podemos obtener resultados de los landmarks
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id,lm)
            #Obtener el punto pixel donde esta
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)),(70, 50),cv.FONT_HERSHEY_PLAIN,3,(255, 0, 0), 3)
    cv.imshow("Video",img)
    cv.waitKey(1)





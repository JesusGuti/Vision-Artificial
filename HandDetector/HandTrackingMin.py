import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
#Vamos a crear un modulo Mano
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

#Obtener Webcam
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #Obtenemos la informacion de cada mano
    if results.multi_hand_landmarks:
        #Para cada mano
        for handLm in results.multi_hand_landmarks:
            for id, lm in enumerate(handLm.landmark):
                #Obtenemos el punto con sus posiciones respectivas en decimales
                print(id,lm)
                #Obtendremos la altura, el ancho y el canal de nuestra imagen
                h,w,c = img.shape
                #Obtenemos la posicion en pixeles
                cx,cy = int(lm.x*w,), int(lm.y*h)
                #print(id,cx,cy)
                cv2.circle(img,(cx,cy),15,(0,224,255),cv2.FILLED)

            #NO el RGB sino el original, muestra los puntos de cada mano y las lineas
            mpDraw.draw_landmarks(img,handLm,mpHands.HAND_CONNECTIONS)
    #Calcular FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

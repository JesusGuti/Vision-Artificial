import cv2
import numpy as np
import HandDetectorModule2 as htm
import time
import autopy


wCam, hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.handDetect(maxHands=1)
wScr, hScr = autopy.screen.size()
frameR = 100
smooth = 10
plocX,plocY = 0, 0
clocX,clocY = 0, 0


while True:
    # 1. Encontrar los landmarks de la mano
    sucess, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Obtener la punta de los dedos indice y medio
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)

    # 3. Ver que dedos estan levantados
    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    #print(fingers)
    # 4. Solo el dedo Indice : Modo Mover
    if fingers[1] == 1 and fingers[2] == 0:
    # 5. Convertir coordenadas

        x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
        y3 = np.interp(y1,(frameR, hCam-frameR), (0,hScr))

    # 6. Valores suavizados: Para evitar que el mouse se mueva tanto
        clocX = plocX + (x3-plocX)/smooth
        clocY = plocY + (y3-plocY)/smooth
    # 7. Mover el mouse
        autopy.mouse.move(wScr-clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
    # 8. Ambos dedos levantados: Modo Click
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Encontrar distancia entre dedos
        length, img, lineInfo = detector.findDistance(8,12,img)
        print(length)
        # 10. Click mouse si la distancia es corta
        if length < 40:
            cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    #11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    #12. Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)

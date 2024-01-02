import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon=0.5,):
        self.minDetectionCon = minDetectionCon
        #Modulo para deteccion facial
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self,img,draw = True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(results)

        bboxes = []

        #Procesa cara a cara
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih), \
                      int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id,bbox,detection.score])
                if draw:
                    self.fancyDraw(img,bbox)

                cv.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                           cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        return img,bboxes

    def fancyDraw(self,img,bbox,l=30,t=10):
        x, y, w, h = bbox
        x1, y1 = x+w,y+h
        cv.rectangle(img, bbox, (88, 240, 134), 1)
        #Esquina Izquierda Superior
        cv.line(img,(x,y),(x+l,y),(88, 240, 134),t)
        cv.line(img, (x, y), (x , y+l), (88, 240, 134), t)
        #Esquina Derecha Superior
        cv.line(img, (x1, y), (x1 - l, y), (88, 240, 134), t)
        cv.line(img, (x1, y), (x1, y + l), (88, 240, 134), t)
        # Esquina Izquierda Inferior
        cv.line(img, (x, y1), (x + l, y1), (88, 240, 134), t)
        cv.line(img, (x, y1), (x, y1 - l), (88, 240, 134), t)
        # Esquina Derecha Inferior
        cv.line(img, (x1, y1), (x1 - l, y1), (88, 240, 134), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (88, 240, 134), t)
        return img

def main():
    # Con webcam, pero hay un limite de Frames
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        print(bboxes)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("Video", img)
        cv.waitKey(1)

if __name__=="__main__":
    main()
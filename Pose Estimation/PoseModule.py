import cv2 as cv
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, complexity=1, smooth=True, segmentation=True, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self,img,draw = True):
        # Debemos convertir la imagen a RGB
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    #Metodo para reescalar video
    def rescaleFrame(self,frame,scale = 0.30):
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation = cv.INTER_AREA)

    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id,lm)
                #Obtener el punto pixel donde esta
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)
        return lmList

def main():
    # Obtenemos el video
      cap = cv.VideoCapture('PoseVideos/2.mp4')
      pTime = 0
      detector = poseDetector()
      while True:
          sucess, img = cap.read()
          frame_resized = detector.rescaleFrame(img)
          frame_resized = detector.findPose(frame_resized)
          imList = detector.findPosition(frame_resized)
          if len(imList) != 0:
            print(imList)
          cTime = time.time()
          fps = 1 / (cTime - pTime)
          pTime = cTime
          cv.putText(frame_resized, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
          cv.imshow("Video", frame_resized)
          cv.waitKey(1)
if __name__ == "__main__":
        main()
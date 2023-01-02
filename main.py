import cv2
import mediapipe
import cvzone
import time
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
listimg = os.listdir("Images")
imgList = []
for imagepath in listimg:
 image = cv2.imread(f'Images/{imagepath}')
 imgList.append(image)

indexImg = 0
#img1 = cv2.imread('Images\')

segmentor = SelfiSegmentation()
pTime = 0
while True:
    success,img = cap.read()
    imgout = segmentor.removeBG(img,imgList[indexImg],threshold= 0.9)
    imagestacked = cvzone.stackImages([img,imgout],2,1)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    imagestacked=cv2.putText(imagestacked,f'FPS:{str(fps)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color = (255, 0, 0),thickness = 2)
    cv2.imshow('Img:',imagestacked)

    #cv2.imshow('Img:', img)
    #cv2.imshow('Img:', imgout)
    key=cv2.waitKey(1)
    if key==ord('a'):
        if indexImg>0:
         indexImg-=1
    elif key==ord('d'):
        if indexImg<len(imgList)-1:
         indexImg+=1
    elif key==ord('q'):
        break

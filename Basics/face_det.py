import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#face detect in given image
# img = cv2.imread('girl_img.png')
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #importing the cascade xml for detecting face.
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# faces = faceCascade.detectMultiScale(img1, 1.1, 4)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
#
# cv2.imshow('Girl Image', img)
# cv2.waitKey(0)

#face detect live webcam
fWidth = 640
fHeight = 800
cap = cv2.VideoCapture(0)
cap.set(3,fWidth)
cap.set(4,fHeight)
cap.set(10,150)
while True:
    succ, img = cap.read()
    facesDet = faceCascade.detectMultiScale(img,1.1,4)
    for (x, y, w, h) in facesDet:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
    cv2.imshow('Face Det', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
import cv2
import numpy as np


def get_contours(img1):
    _,contours,hierarchy = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        cv2.drawContours(imgClone, cnt, -1, (255, 0, 0), 2)
        peri = cv2.arcLength(cnt, True)
        print(peri)
        app = cv2.approxPolyDP(cnt, 0.02*peri, True)
        print(len(app))
        objCor = len(app)
        x, y, w, h = cv2.boundingRect(app)
        cv2.rectangle(imgClone,(x,y),(x+w,y+h),(0,0,255),3)

img = cv2.imread('cvlogo.png')
imgClone = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
imgStack = np.hstack((imgGray, imgBlur, imgCanny))

get_contours(imgCanny)

#
# cv2.imshow('Orig', img)
# cv2.imshow('Gray', imgGray)
# cv2.imshow('Blur', imgBlur)
# cv2.imshow('Canny', imgCanny)
cv2.imshow('Stacked', imgStack)
cv2.imshow('Cont', imgClone)
cv2.waitKey(0)
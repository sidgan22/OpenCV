import cv2
import numpy as np

def hmin(evt):
    pass
def hmax(evt):
    pass
def smin(evt):
    pass
def smax(evt):
    pass
def vmin(evt):
    pass
def vmax(evt):
    pass
#
# path = 'lambo.png'
# img = cv2.imread(path)

fWidth = 640
fHeight = 800
cap = cv2.VideoCapture(1)
cap.set(3,fWidth)
cap.set(4,fHeight)
cap.set(10,150)
# while True:
#     succ, img = cap.read()
#     cv2.imshow('Col Det', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#Create trackbars to adjust Hue, Saturation
cv2.namedWindow("HSV_Bars")
cv2.resizeWindow("HSV_Bars", 300, 300)
cv2.createTrackbar('Hue Min', 'HSV_Bars', 0, 179, hmin)
cv2.createTrackbar('Hue Max', 'HSV_Bars', 0, 179, hmax)
cv2.createTrackbar('Sat Min', 'HSV_Bars', 0, 255, smin)
cv2.createTrackbar('Sat Max', 'HSV_Bars', 0, 255, smax)
cv2.createTrackbar('Val Min', 'HSV_Bars', 0, 255, vmin)
cv2.createTrackbar('Val Max', 'HSV_Bars', 0, 255, vmax)

while True:
    succ, img = cap.read()
    cv2.imshow('Col Det', img)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hmin_val = cv2.getTrackbarPos('Hue Min', 'HSV_Bars')
    hmax_val = cv2.getTrackbarPos('Hue Max', 'HSV_Bars')
    smin_val = cv2.getTrackbarPos('Sat Min', 'HSV_Bars')
    smax_val = cv2.getTrackbarPos('Sat Max', 'HSV_Bars')
    vmin_val = cv2.getTrackbarPos('Val Min', 'HSV_Bars')
    vmax_val = cv2.getTrackbarPos('Val Max', 'HSV_Bars')
    print('Hue Min: ',hmin_val,'Hue Max: ',hmax_val,'Sat Min: ',smin_val,
          'Sat max: ',smax_val,'Val Min: ',vmin_val,'Val Max: ',vmax_val,)

    #Create a mask
    lower = np.array([hmin_val, smin_val, vmin_val])
    upper = np.array([hmax_val, smax_val, vmax_val])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgRes = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow('Masked', mask)
    cv2.imshow('Final', imgRes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
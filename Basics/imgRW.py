import numpy as np
import cv2

#  add parameter 0 for grayscale and
#  parameter 1 or above for normal 3 channel
def imgLoad():
    image= cv2.imread('cvlogo.png') 
    cv2.imshow("Title",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#converts color png to b&w jpg
def imgWrite():
    print("Converting to b&w")
    img=cv2.imread('cvlogo.png',0)
    img_dup=cv2.imwrite("newImage.png",img)
    cv2.imshow("NewImage",img_dup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    print("Start")
    imgLoad()
    imgWrite()
    print("End")
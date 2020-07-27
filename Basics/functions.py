import cv2
import numpy as np

class BasicFunctions:
    def __init__(self,cr):
        self.cr=True

    def img_show(self, fname, title):
        img = cv2.imread(fname)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def conv_bw(self,fname):
        img = cv2.imread(fname)
        imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('BW', imgGrayScale)
        cv2.waitKey(0)

    def add_blur(self,fname):
        img = cv2.imread(fname)
        imgBlur = cv2.GaussianBlur(img,(5,5),0)
        cv2.imshow('normal', img)
        cv2.imshow('blurred', imgBlur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cap_video(self,srcname):
        # Video capture could include a video or webcam feed
        cap = cv2.VideoCapture(0)
        # set height and width of cap
        cap.set(3, 1000)
        cap.set(4, 400)

        # brightness set
        cap.set(10, 50)

        while True:
            success, img = cap.read()
            print(success)
            cv2.imshow('Video Cap', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def canny_edge(self,fname):
        img = cv2.imread(fname)
        imgcanny = cv2.Canny(img,150,200)
        cv2.imshow('normal', img)
        #optional
        kernel = np.ones((2,2),np.uint8) #Describing kernel using  ones of custom size;
        imgDil =  cv2 .dilate(imgcanny,kernel,iterations=1)

        cv2.imshow('Canny Edges', imgcanny)

        #To dilate image
        cv2.imshow("Canny Dilate",imgDil)

        #to erode
        imgErode = cv2.erode(imgDil,kernel,iterations=1)
        cv2.imshow("Canny Erode after dilation",imgErode)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #upscale or downscale image with loss
    def resize_img(self,fname,w,h):
        img = cv2.imread(fname)
        imgResize = cv2.resize(img,(w,h))
        cv2.imshow("Resized ", imgResize)
        cv2.waitKey(0)

    def crop_img(self, fname,w1,w2,h1,h2):
        img = cv2.imread(fname)
        imgResize = img[h1:h2,w1:w2]
        cv2.imshow("cropped ", imgResize)
        cv2.waitKey(0)
    def warp_img(self):
        img = cv2.imread("cards.jpg")

        width, height = 250, 350
        pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))

        cv2.imshow("Image", img)
        cv2.imshow("Output", imgOutput)
        cv2.waitKey(0)

    def stack_hv(self):
        img = cv2.imread('cards.jpg')
        imgHor = np.hstack((img, img))
        imgVer = np.vstack((img, img))

        cv2.imshow("Horizontal", imgHor)
        cv2.imshow("Vertical", imgVer)

        cv2.waitKey(0)

    def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                    (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,
                                                    scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],
                                                                                     cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    img = cv2.imread('cards.jpg')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgStack = stackImages(0.5, ([img, imgGray, img], [img, img, img]))

    cv2.imshow("ImageStack", imgStack)

    cv2.waitKey(0)


if __name__ == "__main__" :
    print("Start")
    ob = BasicFunctions(True)
    print("End")
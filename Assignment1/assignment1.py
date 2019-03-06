
'''
Assignment1
Basic video processing with simple filters and tresholding

Anna Pastwa
March 2019

Python3.6.8
openCV 3.4.1
'''

import cv2
print("OpenCV version :  {0}".format(cv2.__version__))
import numpy as np
import glob
import os
from tqdm import tqdm

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.

cap = cv2.VideoCapture('mymovie.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )



def save_frames():
    currentFrame = 0
    while(currentFrame<length):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    #return frames


def frames_to_gray():
    #2.1 part
    for i in range(450):
        name = './data/frame' +str(i) +'.jpg'
        image = cv2.imread(name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(name, gray)
    print("Finished grayscale")

def blur_gauss():
    #2.2 part
    for i in range(150,300):
        name = './data/frame' + str(i) + '.jpg'
        image = cv2.imread(name)
        blur = cv2.GaussianBlur(image,(i-131+i%2,i-131+i%2),0) #sigmaX increasing
        cv2.imwrite(name, blur)
    print("Finished Gaussian blur")

def filter_bilateral():
    #2.3 part
    for i in range(300,450):
        name = './data/frame' + str(i) + '.jpg'
        image = cv2.imread(name)
        bilateral = cv2.bilateralFilter(image, 9, i-265, i-265) #first value takes the neighbourhood
        cv2.imwrite(name, bilateral)
    print("Finished bilateral")


def color_tresholdRGB():
    #2.5
    for i in range(600, 750):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)

        #rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(img, (0, 100, 95), (60, 165, 145))

        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

        cv2.imwrite(name, green)
    print("Finished tresholding color RGB")


def color_treshold():
    #part 2.6
    for i in range(750,900):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        ## convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        mask = cv2.inRange(hsv, (30,25,25), (80,255,255))

        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

        cv2.imwrite(name, green)
    print("Finished tresholding color")

def improved_grabbing():
    #part 2.7
    for i in range(900,1050):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 25, 25), (80, 255, 255))
        kernel = np.ones((5, 5),'int')
        erosion = cv2.erode(mask, kernel, iterations=1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

        ## slice the green
        fg = cv2.bitwise_or(img, img, mask=opening)

        cv2.imwrite(name, fg)
    print("Finished improved grabbing")

def improved_grabbing2():
    #add 2.8
    for i in range(1300,1500):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 25, 25), (80, 255, 255))
        kernel = np.ones((5, 5),'int')
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        fg = cv2.bitwise_or(img, img, mask=opening)
        ret, thrshed = cv2.threshold(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = contours[4]
            cv2.drawContours(fg, [cnt], 0, (0, 255, 0), 3)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    cv2.rectangle(fg, (5, 40), (400, 100), (0, 255, 255), -2)
                    cv2.putText(fg, 'Green Object Detected', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

                cv2.imwrite(name, fg)
        else:
            print("Sorry No contour Found.")
            cv2.imwrite(name, fg)


def change_color():
    #add 2.9
    for i in range(1050,1300):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 25, 25), (80, 255, 255))
        kernel = np.ones((5, 5),'int')
        erosion = cv2.erode(mask, kernel, iterations=1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

        ## slice the green
        fg = cv2.bitwise_or(img, img, mask=opening)
        fg[mask == 255] = [0, 0, 255]
        cv2.imwrite(name, fg)
    print("Finished changing color")

def change_color_color():
    #add 2.9
    for i in range(1500,1700):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 25, 25), (80, 255, 255))
        kernel = np.ones((5, 5),'int')
        erosion = cv2.erode(mask, kernel, iterations=1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

        ## slice the green
        fg = cv2.bitwise_not(img, img, mask=opening)
        fg[mask == 255] = [0, 0, 255]
        cv2.imwrite(name, fg)
    print("Finished changing color on background")

def fadeaway():
    a = 1
    for i in range(1700,1832):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)


        rows, cols, channels = img.shape
        M = np.float32([[1, 0, a], [0, 1, a]])
        a = 5+a
        dst = cv2.warpAffine(img_edge, M, (cols, rows))
        cv2.imwrite(name, dst)
    print("Finished fading away")

def save():
    os.system("ffmpeg -f image2 -r 30 -i ./data/frame%01d.jpg -vcodec mpeg4 -y anothermovie.mp4")
    os.system("ffmpeg -i anothermovie.mp4 anothermovie.mpeg")

if __name__ == '__main__':

    save_frames()
    frames_to_gray()
    blur_gauss()
    color_tresholdRGB()
    color_treshold()
    improved_grabbing()
    improved_grabbing2()
    change_color()
    change_color_color()
    fadeaway()
    #save_vid_from_jpg()
    save()
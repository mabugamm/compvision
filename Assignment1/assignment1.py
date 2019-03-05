
'''
Assignment1
Basic video processing

Python3.6.8
'''

import cv2
print("OpenCV version :  {0}".format(cv2.__version__))
import numpy as np
import glob

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.

cap = cv2.VideoCapture('mymovie.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

frames = []

def save_frames():
    currentFrame = 0
    while(currentFrame<length):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        name2 = 'frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        frames.append(name2)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def save_vid_from_jpg():
    img_array = []
    for filename in glob.glob('./data/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project.MPEG',cv2.VideoWriter_fourcc(*'mpeg'), 27, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



def frames_to_gray():
    for i in range(405):
        image = cv2.imread('./data/'+frames[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./data/'+frames[i], gray)

if __name__ == '__main__':

    save_frames()
    frames_to_gray()
    save_vid_from_jpg()
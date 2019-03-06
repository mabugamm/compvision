import cv2
import matplotlib as plt

cap = cv2.VideoCapture('mymovie.mp4')

def play_gray():
    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def to_gray():
    image = cv2.imread('./data/frame135.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (155, 155), 0)
    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)
    cv2.imshow('Gaussian blur image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tresholds():
    img = cv2.imread('./data/frame500.jpg', 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

def save_vid_from_jpg():

    img_array = []
    for filename in glob.glob('./data/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project.MPEG',cv2.VideoWriter_fourcc(*'mpeg'), 30, size)

    #TRANSFORM TO ONE GIANT LOOP!!!
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def show(img):
    cv2.imshow('Gaussian blur image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def improved_grabbing():
    for i in range(900,910):
        name = './data/frame' + str(i) + '.jpg'
        img = cv2.imread(name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 25, 25), (80, 255, 255))
        kernel = np.ones((5, 5),'int')
        dilated = cv2.dilate(mask, kernel)
        res = cv2.bitwise_and(img, img, mask=mask)
        ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[4]
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cv2.putText(img, 'Green Object Detected', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(img, (5, 40), (400, 100), (0, 255, 255), 2)

            cv2.imwrite(name, img)
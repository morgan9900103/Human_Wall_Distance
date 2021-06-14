import numpy as np
import sys
import cv2


def process_image(img, upper=4, lower=1):
    height, width = np.shape(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] > upper:
                img[i][j] = 255
            elif img[i][j] < lower:
                img[i][j] = 0
            else:
                img[i][j] = 127

    img = np.uint8(img)
    return img


def find_clearance(img):
    img = np.uint8(img)
    avg = np.mean(img)
    img_canny = cv2.Canny(img, avg - 0.25*avg, avg + 0.25*avg)

    height, width = np.shape(img)
    roi = np.zeros(shape=(height, width), dtype=np.uint8)
    roi[30:108, 60:120] = img_canny[30:108, 60:120]

    img_contour, contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    approx = [None]*len(contours)
    bound_rect = [None]*len(contours)

    for i, cnt in enumerate(contours):
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx[i] = cv2.approxPolyDP(cnt, epsilon, True)
        bound_rect[i] = cv2.boundingRect(approx[i])

    left = bound_rect[0][0] - 60

    right = 120 - (bound_rect[0][0] + bound_rect[0][2])

    if left > right:
        print("left", left*1.5/60)
    else:
        print("right", right*1.5/60)


if __name__ == '__main__':
    file = sys.argv[1]
    depth_img = np.loadtxt(file)
    img_processed = process_image(depth_img)
    find_clearance(img_processed)

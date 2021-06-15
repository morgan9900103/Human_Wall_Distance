import numpy as np
import sys
import cv2


def process_image(img, human_dist=2):
    """
    This function will filter out the data points which are not in the same distance of human (assume 2m)

    return: img
    """
    height, width = np.shape(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] > human_dist+2:
                img[i][j] = 255
            elif img[i][j] < human_dist-1:
                img[i][j] = 0
            else:
                img[i][j] = 127

    img = np.uint8(img)
    visual = True
    if visual:
        img_re = cv2.resize(img, (176*5, 132*5))
        cv2.imshow("img", img_re)
        cv2.waitKey()
    return img


def find_clearance(img, human_dist=2, corridor=1.5):
    """
    This function will find the clearance of human/shelf or human/wall. We will assume the corridor is 1.5m wide
    """
    # Canny edge detection
    avg = np.mean(img)
    threshold1 = 0.8 * avg
    threshold2 = 1.2 * avg
    img_canny = cv2.Canny(img, threshold1, threshold2)
    visual = False
    if visual:
        img_canny_re = cv2.resize(img_canny, (176*5, 132*5))
        cv2.imshow("img_canny", img_canny_re)
        cv2.waitKey()

    # Define size of roi
    height, width = np.shape(img)
    roi = np.zeros(shape=(height, width), dtype=np.uint8)
    h1 = 0      # ceiling
    h2 = 106    # ground
    w1 = 60     # shelf
    w2 = 122    # wall
    roi[h1:h2, w1:w2] = img_canny[h1:h2, w1:w2]
    visual = True
    if visual:
        roi_re = cv2.resize(roi, (176*5, 132*5))
        cv2.imshow("roi", roi_re)
        cv2.waitKey()

    # Find Contour
    img_contour, contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    visual = False
    if visual:
        img_contour_re = cv2.resize(img_contour, (176*5, 132*5))
        cv2.imshow("img_contour", img_contour_re)
        cv2.waitKey()

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    approx = [None]*len(contours)
    bound_rect = [None]*len(contours)
    print("len(contours", len(contours))

    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx[i] = cv2.approxPolyDP(cnt, epsilon, True)
        bound_rect[i] = cv2.boundingRect(approx[i])
        # print("bound_rect[i]", bound_rect[i])

    left = (bound_rect[0][0] - w1) * corridor / (w2 - w1)

    right = (w2 - (bound_rect[0][0] + bound_rect[0][2])) * corridor / (w2 - w1)

    if left > right:
        print("left", left)
    else:
        print("right", right)


if __name__ == '__main__':
    file = sys.argv[1]
    depth_img = np.loadtxt(file)
    img_processed = process_image(depth_img)
    find_clearance(img_processed)

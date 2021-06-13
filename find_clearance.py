import numpy as np
import sys


def process_image(img, lower=1, upper=4):
    height, width = np.shape(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] > upper:
                img[i][j] = 255
            elif img[i][j] < lower:
                img[i][j] = 0
            else:
                img[i][j] = 127

    return img


if __name__ == '__main__':
    file = sys.argv[1]
    depth_img = np.loadtxt(file)
    img_processed = process_image(depth_img)

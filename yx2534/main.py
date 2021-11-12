

import numpy as np
import matplotlib.pyplot as plt
import math
import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage import filters
import matplotlib as mpl
import cv2 as cv


MAX_GREYSCALE = 255

def import_im(filename):
    # Import image as grayscale by arg = 0
    im = cv.imread(filename, 0) * 1.0

    # im_uint8 = ((im - np.min(im)) * (1/(np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')
    return im


def convert_im_grayscale(im):
    return ((im - np.min(im)) * (1 / (np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')


def Farneback_OF(im1, im2):
    optical_flow = cv.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u =

def trajectory_tracking(prev_im, curr_im, time, Ix, Iy):
    row, col = prev_im.size
    new_im_pixels = np.zeros(1, 3);
    optical_flow = cv.calcOpticalFlowFarneback(prev_im, curr_im, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    for i in range (0, row):
        for j in range (0, col):
            if prev_im(i, j) == 0:
                continue



class Track:
    def __init__(self, coord, label, is_stopped, init_time, gray_value):
        self.coord = coord
        self.label = label
        self.is_stopped = is_stopped
        self.init_time = init_time
        self.gray_value = gray_value


def main():
    img_1 = import_im('eig_marple13_01.jpg')
    img_2 = import_im('eig_marple13_02.jpg')
    row, col = img_1.shape

    img_1 = cv.blur(img_1, (5, 5))
    img_2 = cv.blur(img_2, (5, 5))

    Ix1 = filters.sobel_v(img_1)
    Iy1 = filters.sobel_h(img_1)

    Ix2 = filters.sobel_v(img_2)
    Iy2 = filters.sobel_v(img_2)


    neighbor_distance = np.array([
        [[-1, -1], [0, -1], [1, -1]],
        [(-1, 0), [0, 0], [1, 0]],
        [[-1, 1], [0, 1], [1, 1]]
    ])
    '''
    optical_flow = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vector_mag = img_1
    for i in range (0, row):
        for j in range(0, col):
            vector_mag[i, j] = np.sqrt(optical_flow[i, j][0] ** 2 + optical_flow[i, j][1] ** 2)
    
    print (optical_flow[4, 1][0])        
    print(vector_mag) 
    '''


    print (row, " ", col)

    fig1 = plt.figure(figsize=(14, 7))
    plt.axis('off')
    plt.imshow(img_1, cmap='gray')
    plt.show()
    fig2 = plt.figure(figsize=(14, 7))
    plt.axis('off')
    plt.imshow(img_2, cmap='gray')
    plt.show()
    # h, w = img_1.shape
    # print(h)
    # print(w)
    # for row in range(0, h):
    #     for col in range(0, w):
    #         test = img_1[col, row]



if __name__ == "__main__":
    main()
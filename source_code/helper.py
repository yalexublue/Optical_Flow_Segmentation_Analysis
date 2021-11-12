import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def import_im(filename):
    im = cv.imread(filename, 0) * 1.0
    return im

def display_im(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def get_derivative_x(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)

def get_derivative_y(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

def get_neighborhood(row_index: int, col_index: int, ksize: int = 3):
    radius = np.floor(ksize / 2).astype(int)
    col_low = col_index - radius
    col_high = col_index + radius
    row_low = row_index - radius
    row_high = row_index + radius

    neighborhood = []

    for row in range(row_low, row_high + 1):
        for col in range(col_low, col_high + 1):
            neighborhood.append((row, col))

    return neighborhood

def get_gaussian_filter(size=5, sigma=1.0):
    kernel_1d = cv.getGaussianKernel(size, sigma, cv.CV_32F)
    kernel_1d = np.array(kernel_1d)
    kernel_2d = np.dot(kernel_1d, np.transpose(kernel_1d))
    return kernel_2d

def calculate_forward_flow(frames):
    output_flow = np.ones_like(frames[0])
    for i in range(0, len(frames)-1):
        output_flow = cv.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return output_flow


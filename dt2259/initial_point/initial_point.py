#!/usr/bin/python3
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

MAX_GREYSCALE = 255

#Create Gaussian filter
def create_gaussian_filter(sigma):
    # How to choose size n
    n = 2*math.floor(sigma*3)+1
    # Precompute sigma*sigma
    sigma2 = sigma*sigma
    
    # Create a coordinate sampling from -n/2 to n/2 so that (0,0) will be at the center of the filter
    x = np.linspace(-n/2.0, n/2.0, n)
    y = np.linspace(-n/2.0, n/2.0, n)
    
    # Blank array for the Gaussian filter
    gaussian_filter = np.zeros((n,n))

    # Loop over all elements of the filter
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            # Use the x and y coordinate sampling as the inputs to the 2D Gaussian function
            gaussian_filter[i,j] = (1/(2*math.pi*sigma2))*math.exp(-(x[i]*x[i]+y[j]*y[j])/(2*sigma2))
        
    # Normalize so the filter sums to 1
    return gaussian_filter/np.sum(gaussian_filter.flatten())

def import_im(filename):
    # Import image as grayscale by arg = 0
    im = cv.imread(filename, 0) * 1.0

    return im

def convert_im_grayscale(im):
    return ((im - np.min(im)) * (1/(np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')

def blur_img(im_in, sigma_blur):
    g = create_gaussian_filter(sigma_blur)
    blur_im = cv.filter2D(im_in, -1 ,g)
    return blur_im

def vid_partial_derivitive(frames_denoise):
    frames_gradx = []
    frames_grady = []
    for f in frames_denoise:
        gradx = cv.Sobel(f,cv.CV_64F,1,0,ksize=3)
        grady = cv.Sobel(f,cv.CV_64F,0,1,ksize=3)
        
        frames_gradx.append(gradx)
        frames_grady.append(grady)

    return [frames_gradx, frames_grady]

def img_partial_derivitive(img):

    gradx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    grady = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

    return [gradx, grady]

def second_moment_matrix(Ix, Iy, img, threshold):
    h, w = img.shape
    
    neighbor_distance = np.array([
                                [[-1,-1], [0,-1], [1,-1]],
                                [(-1,0), [0,0], [1,0]],
                                [[-1,1], [0,1], [1,1]]
                            ])

    for row in range(1, h-1):
        for col in range(1, w-1):
            point = np.array([col, row])
            M = 0.0
            for row_neighbor in neighbor_distance:
                for item in row_neighbor:
                    neighbor_point = point + item
                    itensity_x = Ix[neighbor_point[1], neighbor_point[0]]
                    itensity_y = Iy[neighbor_point[1], neighbor_point[0]]
                    intensity = np.array([itensity_x, itensity_y])
                    M += intensity * intensity.reshape(2,1)

            eigen_val = np.linalg.eig(M)[0]
            sorted_eigen_val = sorted(eigen_val)

            if sorted_eigen_val[1] < threshold:
                img[row, col] = 0

    return img

def main():
    
    img_1 = import_im('./input_img/frame11.png')

    img_1 = blur_img(img_1, 1)

    img_deriviative = img_partial_derivitive(img_1)

    Ix = img_deriviative[0]
    Iy = img_deriviative[1]
    img = second_moment_matrix(Ix, Iy, img_1, 10000)

    plt.imshow(img, cmap='gray')
    plt.show()

    cv.imwrite('./output_img/frame11_t.png', img)



if __name__ == "__main__":
    main()
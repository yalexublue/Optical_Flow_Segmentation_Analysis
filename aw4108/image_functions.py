import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import math


def open_image(image_path: str):
    image = cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)
    image = ((image - np.min(image))
             * (1 / (np.max(image) - np.min(image)) * 1.0)).astype('float')
    return image


def output_intensity_mapping(image: np.ndarray):
    output_im = np.zeros(image.shape, dtype=np.uint8)
    arr_max = image.max()
    arr_min = image.min() + 1e-10

    for index in range(len(image.ravel())):
        output_im.ravel()[index] = \
            int(np.floor(((image.ravel()[index] - arr_min)
                          / (arr_max - arr_min)) * 255 + 0.5))

    return output_im


def output_image(image: np.ndarray, save_name):
    image_out = output_intensity_mapping(image)
    fig = plot.figure(figsize=(7, 7))
    plot.axis('off')
    plot.imshow(image_out, cmap='gray')
    plot.savefig('./results/' + save_name, bbox_inches='tight')
    plot.show()


def unpack_video(video_path: str, output_file_name: str):
    cam = cv.VideoCapture(video_path)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating director for frames')

    curr_frame = 0

    while True:
        ret, frame = cam.read()

        if ret:
            name = './data/' + output_file_name \
                   + str(curr_frame) + '.jpg'
            print('Creating...' + output_file_name
                  + str(curr_frame) + '.jpg')

            cv.imwrite(name, frame)

            curr_frame += 1
        else:
            break

    cam.release()
    cv.destroyAllWindows()


def zero_padding(num_layers: int, image: np.ndarray):
    return np.pad(image, ((num_layers, num_layers),
                          (num_layers, num_layers)),
                  'constant', constant_values=0)


def get_derivative_x(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)


def get_derivative_y(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)


def threshold_image(image: np.ndarray, thresh: int,
                    below_val: int=0, above_val: int=255):
    out_image = np.zeros(image.shape)
    for row in range(0, image.shape[0]):
        for col in range(0, image.shape[1]):
            if image[row, col] < thresh:
                image[row, col] = below_val
            else:
                image[ row, col] = above_val

    return out_image


def threshold_eigenvalues(image: np.ndarray, thresh: float, window_size: int=3):
    im_ddx = get_derivative_x(image)
    im_ddy = get_derivative_y(image)

    im_eig = np.zeros(image.shape, dtype=np.float32)

    pad = int(np.floor(window_size/2))

    for row in range(pad, image.shape[0]-pad):
        for col in range(pad, image.shape[1]-pad):
            v_matrix = get_second_moment_matrix(row, col, im_ddx, im_ddy, window_size)
            v_eigens = get_eigenvalues(v_matrix)
            v_eigens = sorted(v_eigens)

            if v_eigens[1] < thresh:
                im_eig[row][col] = 0
            else:
                im_eig[row][col] = image[row][col]

    return im_eig


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


def display_image(image: np.ndarray):
    image = output_intensity_mapping(image)
    # threshold_image(image, 145)

    fig = plot.figure(figsize=(7, 7))
    plot.imshow(image, cmap='gray')
    plot.show()


def get_flow_magnitude_array(op_flow: np.ndarray) -> np.ndarray:
    mag = np.zeros((op_flow.shape[0], op_flow.shape[1]))
    for row in range(0, op_flow.shape[0]):
        for col in range(0, op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            mag[row][col] = np.sqrt((u*u) + (v*v))

    return mag


def get_gaussian_filter(size=5, sigma=1.0):
    kernel_1d = cv.getGaussianKernel(size, sigma, cv.CV_32F)
    kernel_1d = np.array(kernel_1d)
    kernel_2d = np.dot(kernel_1d, np.transpose(kernel_1d))
    return kernel_2d


def get_flow_angle_array(op_flow: np.ndarray) -> np.ndarray:
    angles = np.zeros((op_flow.shape[0], op_flow.shape[1]))

    for row in range(0, op_flow.shape[0]):
        for col in range(0, op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            angles[row][col] = np.arctan2(v, u) * (180 / np.pi)

    return angles


def get_second_moment_matrix(coord_row, coord_col, image_ddx: np.ndarray,
                             image_ddy: np.ndarray, window_size: int):
    # Define an gaussian kernel the size of the window and construct window
    window_weight = get_gaussian_filter(window_size)
    neighborhood = get_neighborhood(coord_row, coord_col, window_size)
    neighborhood = np.reshape(neighborhood, (window_size, window_size, 2))

    second_moment_matrix = np.zeros((2, 2), dtype=np.float32)

    for row in range(0, window_size):
        for col in range(0, window_size):
            p_row = neighborhood[row][col][0]
            p_col = neighborhood[row][col][1]
            ddx = image_ddx[p_row][p_col]
            ddy = image_ddy[p_row][p_col]

            ddx = ddx * ddx
            ddxy = ddx * ddy
            ddy = ddy * ddy

            weight = window_weight[row][col]

            second_moment_matrix[0][0] += weight * ddx
            second_moment_matrix[0][1] += weight * ddxy
            second_moment_matrix[1][0] += weight * ddxy
            second_moment_matrix[1][1] += weight * ddy

    return second_moment_matrix


def get_eigenvalues(matrix: np.ndarray):
    eigs = np.linalg.eig(matrix)
    return eigs[0]


def bilinear_interpolation(off_grid_point_row, off_grid_point_col, array: np.ndarray):
    row_weight = off_grid_point_row - np.floor(off_grid_point_row)
    col_weight = off_grid_point_col - np.floor(off_grid_point_col)

    if off_grid_point_row < 0 or off_grid_point_row + 1 >= array.shape[0] \
        or off_grid_point_col < 0 or off_grid_point_col + 1 >= array.shape[1]:
        return np.nan

    u_left = array[np.floor(off_grid_point_row).astype(int)][np.floor(off_grid_point_col).astype(int)]
    u_right = array[np.floor(off_grid_point_row).astype(int)][np.ceil(off_grid_point_col).astype(int)]
    l_left = array[np.ceil(off_grid_point_row).astype(int)][np.floor(off_grid_point_col).astype(int)]
    l_right = array[np.ceil(off_grid_point_row).astype(int)][np.ceil(off_grid_point_col).astype(int)]

    interpolated_value = u_left * (1-row_weight) * (1-col_weight) + \
                         u_right * col_weight * (1-row_weight) + \
                         l_left * (1-col_weight) * row_weight + \
                         l_right * col_weight * row_weight

    return interpolated_value

def out_of_bounds(point: tuple, image_shape: tuple) -> bool:
    status = False

    if point[0] < 0 or point[0] >= image_shape[0] \
        or point[1] < 0 or point[1] >= image_shape[1]:
        status = True

    return status
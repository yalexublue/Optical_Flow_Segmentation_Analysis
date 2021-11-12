import numpy as np
import helper

def get_second_moment_matrix(coord_row, coord_col, image_ddx: np.ndarray,
                             image_ddy: np.ndarray, window_size: int):
    # Define an gaussian kernel the size of the window and construct window
    window_weight = helper.get_gaussian_filter(window_size)
    neighborhood = helper.get_neighborhood(coord_row, coord_col, window_size)
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

def threshold_eigenvalues(image: np.ndarray, thresh: float, window_size: int=3):
    im_ddx = helper.get_derivative_x(image)
    im_ddy = helper.get_derivative_y(image)

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
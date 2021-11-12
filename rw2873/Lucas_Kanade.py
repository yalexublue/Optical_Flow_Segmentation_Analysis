import image_functions
import numpy as np


def flow(image_1: np.ndarray, image_2: np.ndarray,
         aperture_size: int, blobby: bool) -> np.ndarray:
    # This is the driver function for the Lucas-Kanade program.
    # Users should include Lucas_Kanade.py in their import list,
    # after, calls can be made to the optical flow pipeline by
    # calling Lucas_Kana.flow() and passing the appropriate
    # arguments into the function.  Assumes the image is already
    # opened as a numpy array in the calling program.

    # PRECONDITION: it is assumed by the program that frame 1 is
    #               at time-step 't' and frame 2 is at time-step
    #               't+1'.

    '''
    Pipeline steps:
        Sobel derivatives of x and y
            (includes Gaussian smoothing)
        Gradient Magnitude on 1st derivs
        delta_time = frame (t) - frame(t+1)
        Brightness constancy = gm_x * u + gm_y * v + delta_time = 0
        Define an aperture window (5x5)
        Populate a linear system for each pixel in neighborhood
        Compute optical flow score
        Place into optical flow image array
    '''

    APERTURE_SIZE = aperture_size
    INFORMATION_LIMIT = np.floor(APERTURE_SIZE / 2).astype(int)

    # Smooth and get 1st derivatives for both images (uses 3x3 Sobbel)
    im_1_deriv_x = image_functions.get_derivative_x(image_1)
    im_1_deriv_y = image_functions.get_derivative_y(image_1)

    # Calculate forward difference between the frames
    im_deriv_time = image_1 - image_2

    optical_flow = np.zeros((image_1.shape[0], image_1.shape[1], 2))

    for row in range(INFORMATION_LIMIT,
                     image_1.shape[0] - INFORMATION_LIMIT,
                     INFORMATION_LIMIT):
        for col in range(INFORMATION_LIMIT,
                         image_1.shape[1] - INFORMATION_LIMIT,
                         INFORMATION_LIMIT):
            # Initialize known values for point
            # partial_x = im_1_deriv_x[row][col]
            # partial_y = im_1_deriv_y[row][col]
            # deriv_time = im_deriv_time[row][col]

            # Solve for disparity parameters u and v by least squares
            neighborhood_x = im_1_deriv_x[row-INFORMATION_LIMIT:row+INFORMATION_LIMIT+1,
                             col-INFORMATION_LIMIT:col+INFORMATION_LIMIT+1].reshape(-1, 1)
            neighborhood_y = im_1_deriv_y[row-INFORMATION_LIMIT:row+INFORMATION_LIMIT+1,
                             col-INFORMATION_LIMIT:col+INFORMATION_LIMIT+1].reshape(-1, 1)
            neighborhood_t = im_deriv_time[row-INFORMATION_LIMIT:row+INFORMATION_LIMIT+1,
                             col-INFORMATION_LIMIT:col+INFORMATION_LIMIT+1].reshape(-1, 1) * (-1)
            matrix_a = np.hstack([neighborhood_x, neighborhood_y])

            '''
            # Attempt to implement the matrix multiplication version of least squares
            a = np.sum(np.dot(neighborhood_x, neighborhood_x))
            b = np.sum(np.dot(neighborhood_x, neighborhood_y))
            c = np.sum(np.dot(neighborhood_y, neighborhood_x))
            d = np.sum(np.dot(neighborhood_y, neighborhood_y))
            e = np.sum(np.dot(neighborhood_x, neighborhood_t)) * -1
            f = np.sum(np.dot(neighborhood_y, neighborhood_t)) * -1
            
            matrix_a = np.array([[a, b], [c, d]])
            matrix_b = np.array([[e], [f]])
            '''
            # Solve for unknown parameters with least squares
            op_flow = np.linalg.lstsq(matrix_a, neighborhood_t, rcond=None)
            op_flow = op_flow[0]

            if blobby:
                # Calculate optical flow value for the aperture patch
                for sub_row in range(row-INFORMATION_LIMIT, row+INFORMATION_LIMIT + 1):
                    for sub_col in range(col - INFORMATION_LIMIT, col + INFORMATION_LIMIT + 1):
                        optical_flow[sub_row][sub_col][0] = op_flow[0]  # u value
                        optical_flow[sub_row][sub_col][1] = op_flow[1]  # v value
            else:
                optical_flow[row][col][0] = op_flow[0]
                optical_flow[row][col][1] = op_flow[1]

    return optical_flow

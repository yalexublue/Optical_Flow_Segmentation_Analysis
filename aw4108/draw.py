import image_functions
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plot
import cv2 as cv


def draw_flow_intensity(op_flow: np.ndarray):
    out_im = np.zeros((op_flow.shape[0], op_flow.shape[1]))

    for row in range(0,op_flow.shape[0]):
        for col in range(0,op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            mag = np.sqrt((u*u) + (v*v))

            out_im[row][col] = mag

    return out_im


def draw_flow_arrows(image: np.ndarray, op_flow: np.ndarray, aperture: int, scale: int):
    '''
    op_flow = (row, col, (u,v))

    op_flow_u = u {} key = (row, col) val = u
    '''

    fig = plot.figure(figsize=(7, 7))
    plot.imshow(image, cmap='gray')
    for row in range(0, op_flow.shape[0], aperture):
        for col in range(0, op_flow.shape[1], aperture):
            plot.quiver(col, row, op_flow[row, col, 0], op_flow[row, col, 1], scale=scale, color='blue')
    plot.show()


def draw_flow_hsv(op_flow: np.ndarray) -> np.ndarray:
    hsv_output = np.zeros((op_flow.shape[0], op_flow.shape[1], 3), dtype='float32')
    hsv_output[..., 1] = 255

    magnitudes = image_functions.get_flow_magnitude_array(op_flow)
    magnitudes = image_functions.output_intensity_mapping(magnitudes)

    angles = image_functions.get_flow_angle_array(op_flow)

    for row in range(0, hsv_output.shape[0]):
        for col in range(0, hsv_output.shape[1]):
            hsv_output[row][col][0] = angles[row][col]
            hsv_output[row][col][2] = magnitudes[row][col]

    im_bgr = cv.cvtColor(hsv_output, cv.COLOR_HSV2BGR)

    return im_bgr


def draw_trajectory(image: np.ndarray, trajectories, start_frame: int=0,
                    end_frame: int=1, aperture: int=5, scale: int=100):
    '''
    op_flow = (row, col, (u,v))

    op_flow_u = u {} key = (row, col) val = u
    '''

    fig = plot.figure(figsize=(7, 7))
    plot.imshow(image, cmap='gray')

    # This is the ugliest forloop I've written since I first learned c++.
    # Not enough coffee to come up with something better. Send help.
    for t in range(0, len(trajectories)):
        for frame in range(start_frame, end_frame, aperture):
            for node in range(0, len(trajectories[t].history)):

                if trajectories[t].history[node][2] == frame:
                    row_1 = trajectories[t].history[node][0]
                    col_1 = trajectories[t].history[node][1]

                

                    if node + 1 == len(trajectories[t].history):
                        continue

                    row_2 = trajectories[t].history[node + 1][0]
                    col_2 = trajectories[t].history[node + 1][1]

                    # plot.scatter(col_1,row_1)
                    # plot.scatter(col_2,row_2)

                    # d_row = row_2 - row_1
                    # d_col = col_2 - col_1
                    # plot.quiver(col_1, row_1, d_row, d_col, scale=scale, color='blue')

    '''
    I think I went about this the wrong way.  I tried to draw the 'flow segments' which
    had already been discarded.  I should have simply drawn the points at each frame.
    
    For traj in trajecotires:
        for h in traj.history:
            image_sequence[traj_row][traj_col][traj_frame] == COLOR_DOT
            
    To pull this off, we would need to:
        1) assemble a 3D tensor of image frames from frame 0 to frame n
        2) convert all grayscale to color (keeping their image quality
        3) figure out a color scheme for the dots, if desired.
        
    going to do some AI reading for now.  Will ponder. - Russ
    '''

    plot.show()

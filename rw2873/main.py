#########################################
#   Optical Flow Implementations        #
#   CS-GY 6643, Computer Vision         #
#   Professor James Fishbaugh           #
#   Author:        Russell Wustenberg   #
#   Collaborators: Duc Tran             #
#                  Andrew Weng          #
#                  Ye Xu                #
#########################################
import numpy as np
import image_functions
import draw
import cv2 as cv
import track
import os


# def main():




# '''
def main():
    frame_0 = image_functions.open_image('results/Marple13_eig/eig_marple13_20.jpg')
    frame_0 = image_functions.output_intensity_mapping(frame_0)
    frame_1 = image_functions.open_image('results/Marple13_eig/eig_marple13_21.jpg')
    frame_1 = image_functions.output_intensity_mapping(frame_1)

    trajectories = []
    frames = [frame_0, frame_1]

    frame_dimensions = frames[0].shape

    flow_fore = np.zeros(frame_dimensions, dtype=np.float32)
    flow_back = np.zeros(frame_dimensions, dtype=np.float32)

    for frame in range(0, len(frames)):

        # If the frame is the final frame in the sequence, optical flow cannot
        # be calculated (as there is no frame (t+1)).  In this case, we go throgh
        # all trajectories and set them as 'dead.'  Then we skip out of the loop.
        if frame == len(frames) - 1:
            for trajectory in trajectories:
                trajectory.live = False
            continue
        # Calculate the forward and backwards flow between the current and next frame
        flow_fore = cv.calcOpticalFlowFarneback(frames[frame], frames[frame + 1], flow_fore,
                                                0.5, 5, 5, 5, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow_back = cv.calcOpticalFlowFarneback(frames[frame + 1], frames[frame], flow_back,
                                                0.5, 5, 5, 5, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_back_u, flow_back_v = cv.split(flow_back)

        draw.draw_flow_arrows(frame_0, flow_fore, 20, 100)
        draw.draw_flow_arrows(frame_1, flow_back, 20, 100)
        # Check if new objects are in scene in the FORWARD FLOW
        # At the authors' suggestion, down sample this step by a factor of [4,16]
        # in order to keep trajectory numbers at a minimum.
        for row in range(0, frame_dimensions[0], 4):
            for col in range(0, frame_dimensions[1], 4):

                # check each pixel for flow response
                if abs(flow_fore[row][col][0]) > 1e-20 or abs(flow_fore[row][col][1]) > 1e-20:
                    tracked = False

                    # check each trajectory to see if it is currently tracking that point
                    for trajectory in trajectories:
                        if trajectory.curr_position == (row, col):
                            tracked = True
                            break

                    # Create new trajectory if not currently tracked
                    if not tracked:
                        trajectories.append(track.Track(row, col, frame))

        # For all trajectories, we track them from frame (t) to (t+1).
        for curr_traj in trajectories:
            curr_pos = curr_traj.get_curr_position()

            # sample the forward flow vector at the point's current location
            fwd_flow = flow_fore[curr_pos[0]][curr_pos[1]]

            # New position is (x + u, y + v)
            new_pos = (curr_pos[0] + fwd_flow[1], curr_pos[1] + fwd_flow[0])

            # If the point goes out of frame, kill the trajectory
            if image_functions.out_of_bounds(new_pos, frame_dimensions):
                curr_traj.live = False
                continue

            # The new point is likely 'off the grid' so we sample the backwards flow by
            # bilinear interpolation, as the u and v vectors are orthogonal, they can be
            # processed independently
            bck_flow_u = image_functions.bilinear_interpolation(new_pos[0], new_pos[1], flow_back_u)
            bck_flow_v = image_functions.bilinear_interpolation(new_pos[0], new_pos[1], flow_back_v)

            if bck_flow_u == np.nan or bck_flow_v == np.nan:
                continue

            # We check if the backwards and forwards flow vectors are inverses
            occluded = track.occlusion_detection(fwd_flow, (bck_flow_u, bck_flow_v))
            if occluded:
                # Kill if occluded (or if something went wrong)
                curr_traj.live = False
            else:
                if curr_traj.curr_position[0] - new_pos[0] > 1 or curr_traj.curr_position[1] - new_pos[1] > 1:
                    print("Trajectory at position ", curr_traj.curr_position, "moved to (", new_pos, ')\n')
                # If not occluded, update the point
                curr_traj.set_position(new_pos[0], new_pos[1], frame)

    draw.draw_trajectory(frame_0, trajectories, 0, 1, 5)
    # STEP 3) Construct affinity matrix

    # STEP 4) Populate affinity values

    # Step 5) Spectral Clustering






    return 0
# '''

if __name__ == '__main__':
    main()

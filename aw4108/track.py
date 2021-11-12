import numpy as np


class Track:
    def __init__(self, start_row, start_col, start_frame):
        self.origin = (start_row, start_col, start_frame)
        self.history = []
        self.history.append(self.origin)
        self.curr_position = self.history[0]
        self.live = True
        self.label = -1
        self.trajectory_ddt=[]
        

    def get_history(self):
        return self.history

    def get_origin(self):
        return self.origin

    def get_curr_position(self):
        return self.curr_position

    def get_label(self):
        return self.label

    def set_position(self, row, col, frame):
        new_location = (row, col, frame)
        self.history.append(new_location)
        self.curr_position = new_location

    def set_live(self, status: bool):
        self.live = status

    def set_label(self, label: int):
        self.label = label

    def smooth_trajectory(self, frame_difference: int):


        self.trajectory_ddt=[]

        normalization = 1/frame_difference
        traj_length = len(self.history)


        # address every item in the history
        for hist_index in range(len(self.history)):

            # the current frame we're working with is found at the last index of the history tuple.
            curr_frame = self.history[hist_index][-1]

            # if the remaining number of frames is smaller than the frame window, we need to shrink the frame window.
            # Note: the smallest this window can be is (0), which will return a ddt of 0 (identity). This occurs at the end of the history.
            if (traj_length - 1 ) - hist_index < frame_difference:
                frame_difference = len(self.history) - hist_index - 1
           

            cur_frame_col = self.history[hist_index][1]
            cur_frame_row = self.history[hist_index][0]
            fwd_frame_col = self.history[hist_index + frame_difference][1]
            fwd_frame_row = self.history[hist_index + frame_difference][0]

            fwd_diff_col = normalization * (fwd_frame_col - cur_frame_col)
            fwd_diff_row = normalization * (fwd_frame_row - cur_frame_row)

            self.trajectory_ddt.append((fwd_diff_row, fwd_diff_col, curr_frame))



def calculate_overlap(A: Track, B: Track):
    
    # Grab the start end end frames for the trajectory
    a_start = A.history[0][2]
    a_end = A.history[-1][2]
    b_start = B.history[0][2]
    b_end = B.history[-1][2]

    # check if they overlap. Return overlapping frames, and if no overlap, return (-1,-1)
    a = range(a_start,a_end+1)
    b = range(b_start,b_end+1)
    overlap = list(set(a).intersection(b))

    if overlap == (-1):
        overlap = (-1,-1)


    return overlap


def find_greatest_distance_and_frame(A: Track, B: Track):
    overlap = calculate_overlap(A, B)
    
    # if there's no overlap, we simply return -1.
    if overlap[0] == -1:
        return -1

    # initialize variables for maximum difference in ddt, and the frame at which it occurs
    max_diff = 0
    max_diff_frame = overlap[0]


    # for every frame in which the trajectories overlap...
    for frame in range(overlap[0], overlap[-1]+1):

        # retrieve the corresponding trajectory tuples.
        for a_traj_ddt_tuple in A.trajectory_ddt:
            if a_traj_ddt_tuple[-1] == frame:
                a_ddt = a_traj_ddt_tuple

        for b_traj_ddt_tuple in B.trajectory_ddt:
            if b_traj_ddt_tuple[-1] == frame:
                b_ddt = b_traj_ddt_tuple

        # calculate the difference in row, col
        diff_row = a_ddt[0] - b_ddt[0]
        diff_col = a_ddt[1] - b_ddt[1]

        # take the euclidean distance betwen the ddts
        diff = np.sqrt((diff_row*diff_row) + (diff_col*diff_col))

        # if the difference in ddts is greater, record it and save the frame
        if diff > max_diff:
            max_diff = diff
            max_diff_frame = frame


    return max_diff,max_diff_frame


def occlusion_detection(fwd_opflow: tuple, back_opflow: tuple) -> bool:
    occlusion = False

    ls_sum = (fwd_opflow[0] + back_opflow[0], fwd_opflow[1] + back_opflow[1])
    ls_mag = (ls_sum[0] * ls_sum[0]) + (ls_sum[1] + ls_sum[1])

    w_mag = (fwd_opflow[0] * fwd_opflow[0]) + (fwd_opflow[1] + fwd_opflow[1])
    w_hat_mag = ((back_opflow[0] * back_opflow[0]) + (back_opflow[1] * back_opflow[1]))
    rs_sum = 0.01 * (w_mag + w_hat_mag) + 0.5

    if ls_mag >= rs_sum:
        occlusion = True

    return occlusion



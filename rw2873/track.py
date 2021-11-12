import numpy as np


class Track:
    def __init__(self, start_row, start_col, start_frame):
        self.origin = (start_row, start_col, start_frame)
        self.history = []
        self.history.append(self.origin)
        self.curr_position = self.history[0]
        self.live = True
        self.label = -1

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
        normalization = 1/frame_difference
        traj_length = self.history[-1][2]

        for curr_frame in range(0,traj_length):

            if traj_length - curr_frame < frame_difference:
                frame_difference = self.history[-1][2] - curr_frame

            cur_frame_col = self.history[curr_frame][1]
            cur_frame_row = self.history[curr_frame][0]
            fwd_frame_col = self.history[curr_frame + frame_difference][1]
            fwd_frame_row = self.history[curr_frame + frame_difference][0]

            fwd_diff_col = normalization * (fwd_frame_col - cur_frame_col)
            fwd_diff_row = normalization * (fwd_frame_row - cur_frame_row)

            self.trajectory_ddt.append((fwd_diff_row, fwd_diff_col, curr_frame))


def calculate_overlap(A: Track, B: Track):
    # Grab the start end end frames for the trajectory
    a_start = A.history[0][2]
    a_end = A.history[-1][2]
    b_start = B.history[0][2]
    b_end = B.history[-1][2]

    # Two cases along the time axis
    # A ==============
    # B           ++++++++++++

    # A           ++++++++++++
    # B =============

    # See if they overlap
    if(a_end <= b_start):
        overlap = (b_start, a_end)
    elif(a_start <= b_end):
        overlap = (a_start, b_end)
    else:
        overlap = (-1, -1)

    return overlap


def find_greatest_distance(A: Track, B: Track):
    overlap = calculate_overlap(A, B)

    if overlap[0] == -1:
        return -1

    max_diff = 0
    max_diff_frame = overlap[0]
    for frame in range(overlap[0], overlap[1]+1):
        a_ddt = A.trajectory_ddt[frame]
        b_ddt = B.trajectory_ddt[frame]

        diff_row = a_ddt[0] - b_ddt[0]
        diff_col = a_ddt[1] - b_ddt[1]
        diff = np.sqrt((diff_row*diff_row) + (diff_col*diff_col))

        if diff < max_diff:
            max_diff = diff
            max_diff_frame = frame

    return max_diff_frame


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




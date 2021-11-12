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

def calculate_A(trajectories,gamma):
    # D squared - spatial distance times distance of ddt
    D_sq = np.zeros((len(trajectories),len(trajectories)))

    # D ddt - maximum distance of ddt
    D_ddt = np.zeros((len(trajectories),len(trajectories)))

    # D spatial - spatial distance between points
    D_spat = np.zeros((len(trajectories),len(trajectories)))

    # A - affinity matrix
    A = np.zeros((len(trajectories),len(trajectories)))

    # FIX TO ALLOW FULL MATRIX
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            if i < j: 
                continue

            # for every trajectory pair...
            traj_a = trajectories[i]
            traj_a.smooth_trajectory(2)
            traj_b = trajectories[j]
            traj_b.smooth_trajectory(2)

            # calculate the maximum difference in ddt between the two points
            max_diff, max_diff_frame = track.find_greatest_distance_and_frame(traj_a,traj_b)

            # log a message whenever there's a trajectory that moves.
            if (max_diff > 0.0) and ((i%100 + j%100) == 0):
                print('maximum difference in ddt between tracks',i,'and',j,'is: ',max_diff,' at frame:',max_diff_frame)

            # record the maximum distance between ddts
            D_ddt[i,j] = max_diff

            # for every point's history, we examine the frame where the distance between ddts is maximum
            for a_history_tuple in traj_a.history:
                # print(a_history_tuple)
                if a_history_tuple[-1] == max_diff_frame:
                    a_point = a_history_tuple

            for b_history_tuple in traj_b.history:
                # print(b_history_tuple)
                if b_history_tuple[-1] == max_diff_frame:
                    b_point = b_history_tuple

            # collect the coordinates
            a_coord = np.array([a_point[0],a_point[1]])
            b_coord = np.array([b_point[0],b_point[1]])

            # calculate euclidian (spatial) distance
            euclid_dist = np.sqrt((a_coord - b_coord).dot((a_coord - b_coord).T))
            
            # save spatial distance
            D_spat[i,j] = euclid_dist

            # calculate D_squared, (D_spatial * D_ddt)
            D_sq[i,j] = D_spat[i,j] * D_ddt[i,j]

            A[i,j] = np.exp( (-1 * gamma) * D_sq[i,j] )
            A[j,i] = A[i,j]

    return A

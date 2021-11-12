import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import helper
import threshold
import tracking
import affinity
import spectral_clustering
import image_functions
import os


def main():

    #Step 1: Read in image
    #frame_0 = helper.import_im('../src/images/FBMS_marple13/marple13_20.jpg')
    #frame_1 = helper.import_im('../src/images/FBMS_marple13/marple13_21.jpg')

    #Step 2: Threshold
    #Note: This step take lots of time therefore we get result of this threshold for all 75 frames of maple13 in images/Marple13_eig/
    #threshold_frame_0 = threshold.threshold_eigenvalues(frame_0, 50000)
    #threshold_frame_1 = threshold.threshold_eigenvalues(frame_1, 50000)

    #Hardcode frame
    frame_0 = helper.import_im('../src/images/Marple13_eig/eig_marple13_22.jpg')
    frame_1 = helper.import_im('../src/images/Marple13_eig/eig_marple13_23.jpg')

    frames = [frame_0, frame_1]

    #Grab frame of marple13 in order
    path = '../src/images/Marple13_eig'
    directory = os.fsencode(path)

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            im = helper.import_im(path + '/' + filename)
            #frames.append(im)
        if filename.endswith('20.jpg'):
            break

    #Step 3: Build trajectory and affinity
    trajectories = tracking.create_trajectories(frames)

    A = affinity.calculate_A(trajectories, gamma = 0.1)
    #np.savetxt("A_out.csv", A, delimiter=",")

    #Step 4: Perform spectal clustering
    clustering = spectral_clustering.spectral_clustering(df=A, n_neighbors=3, n_clusters=3)

    for i in range(len(clustering)):
        trajectories[i].label = clustering[i]

    #Display result
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(frame_0, cmap='gray')
    for i in range(len(trajectories)):
        point = trajectories[i].history[0]
        label = trajectories[i].label
        col=point[1]
        row=point[0]
        if label == 0:
            plt.scatter(col,row,c='b')
        if label == 1:
            plt.scatter(col,row,c='y')
        if label == 2:
            plt.scatter(col,row,c='r')
        if label == 3:
            plt.scatter(col,row,c='g')
    plt.show()


if __name__ == "__main__":
    main()
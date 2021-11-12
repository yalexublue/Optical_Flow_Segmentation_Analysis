import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import flow_vis



MAX_GREYSCALE = 255

def import_im(filename):
    # Import image as grayscale by arg = 0
    im = cv.imread(filename, 0) * 1.0
    return im

def flow(frames):
    output_flow = np.ones_like(frames[0])
    for i in range(0, len(frames)-1):
        output_flow = cv.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return output_flow

def euclidean_norm(vector):
    return np.sqrt(vector[0]**2 + vector[1]**2)**2

def partial_derivitive(matrix):
    
    gradx = cv.Sobel(matrix,cv.CV_64F,1,0,ksize=3)
    grady = cv.Sobel(matrix,cv.CV_64F,0,1,ksize=3)
    

    return [gradx, grady]

def inverse_flow(forward_flow, backward_flow):
    h, w, depth = forward_flow.shape
    for rows in range(0, w):
        for cols in range(0, h):
            point_x = [cols, rows]
            curr_forward_vector = forward_flow[cols, rows]
            point_x_w = point_x + curr_forward_vector

            # Get only decimal of number
            point_x_w_dec = [point_x_w[0] - int(point_x_w[0]), point_x_w[1] - int(point_x_w[1])]
            
            # Get distance to reach neighbor point
            distance1 = [point_x_w_dec[0], point_x_w_dec[1]]
            distance2 = [1 - point_x_w_dec[0], 1 - point_x_w_dec[1]]
            distance3 = [-point_x_w_dec[0], 1 - point_x_w_dec[1]]
            distance4 = [1 - point_x_w_dec[0], -point_x_w_dec[1]]
            
            # Get neighborhood point
            point1 = (point_x_w - distance1).astype(int)
            point2 = (point_x_w + distance2).astype(int)
            point3 = (point_x_w + distance3).astype(int)
            point4 = (point_x_w + distance4).astype(int)

            if point1[0] >= h or point1[1] >= w:
                continue
            if point2[0] >= h or point2[1] >= w:
                continue
            if point3[0] >= h or point3[1] >= w:
                continue
            if point4[0] >= h or point4[1] >= w:
                continue
            
            # Get neighborhood vectors
            vector1 = backward_flow[point1[0], point1[1]]
            vector2 = backward_flow[point2[0], point2[1]]
            vector3 = backward_flow[point3[0], point3[1]]
            vector4 = backward_flow[point4[0], point4[1]]

            # Weight (Area)
            w1 = point_x_w_dec[0] * point_x_w_dec[1]
            w2 = 1 - point_x_w_dec[0] * 1 - point_x_w_dec[1]
            w3 = point_x_w_dec[0] * 1 - point_x_w_dec[1]
            w4 = 1 - point_x_w_dec[0] * point_x_w_dec[1]

            # Eclidean norm
            d = euclidean_norm(curr_forward_vector)
            d1 = euclidean_norm(vector1)
            d2 = euclidean_norm(vector2)
            d3 = euclidean_norm(vector3)
            d4 = euclidean_norm(vector4)

            if w1 >= 0.25 and d >= d1:
                backward_flow[point1[0], point1[1]] = -1 * curr_forward_vector

            if w2 >= 0.25 and d >= d2:
                backward_flow[point2[0], point2[1]] = -1 * curr_forward_vector

            if w3 >= 0.25 and d >= d3:
                backward_flow[point3[0], point3[1]] = -1 * curr_forward_vector

            if w4 >= 0.25 and d >= d4:
                backward_flow[point4[0], point4[1]] = -1 * curr_forward_vector

    return backward_flow

def tracking_point_matrix_threshold(flow_forward, flow_inverse):
    h, w, depth = flow_forward.shape
    tracking_point = {}

    derivitive_u = partial_derivitive(flow_forward[::,::, 0])
    derivitive_v = partial_derivitive(flow_forward[::,::, 1])

    for rows in range(0, w):
        for cols in range(0, h):
            curr_point = np.array([cols, rows])
            curr_forward_vector = flow_forward[cols, rows]
            forward_point = np.round(curr_point + curr_forward_vector, 0).astype(int)

            if forward_point[0] < 0  or forward_point[1] < 0:
                continue

            if forward_point[0] >= h or forward_point[1] >= w:
                continue
            
            curr_backward_vector = flow_inverse[forward_point[0], forward_point[1]]
            val1 = euclidean_norm(curr_forward_vector + curr_backward_vector)
            val2 = euclidean_norm(curr_forward_vector)
            val3 = euclidean_norm(curr_backward_vector)

            condition1 = val1 < 0.01 * (val2 + val3) + 0.5

            vector_derivitive_u = [derivitive_u[0][cols, rows], derivitive_u[1][cols, rows]]
            vector_derivitive_v = [derivitive_v[0][cols, rows], derivitive_v[1][cols, rows]]
            
            delta_u = euclidean_norm(vector_derivitive_u)
            delta_v = euclidean_norm(vector_derivitive_v)

            condition2 = delta_u + delta_v <= 0.01 * val2 + 0.002
            
            if condition1 and condition2:
                tracking_point[tuple(curr_point)] = curr_forward_vector

    return tracking_point

def tracking_point_matrix_exact_point(flow_forward, flow_inverse):
    h, w, depth = flow_forward.shape
    tracking_point = {}
    for rows in range(0, w):
        for cols in range(0, h):
            curr_point = np.array([cols, rows])
            curr_forward_vector = flow_forward[cols, rows]
            forward_point = np.round(curr_point + curr_forward_vector, 0).astype(int)

            if forward_point[0] < 0  or forward_point[1] < 0:
                continue

            if forward_point[0] >= h or forward_point[1] >= w:
                continue

            forward_point = forward_point.astype(int)
            backward_point = np.round(forward_point - flow_inverse[forward_point[0], forward_point[1]], 0).astype(int)

            if (curr_point == backward_point).all():
                tracking_point[tuple(curr_point)] = curr_forward_vector

    return tracking_point

def magnitude(size, flow):
    im_mag = np.zeros(size)

    for key, val in flow.items():
        vectors = np.array(list(val))
        points = np.array(list(key))

        u = vectors[0]
        v = vectors[1]

        mag = np.sqrt(u**2 + v**2)

        im_mag[points[0], points[1]] = mag

    return im_mag

def main():
    #img_1 = import_im('./dt2259/occlusion/input_img/eig_marple13_20.jpeg')
    #img_2 = import_im('./dt2259/occlusion/input_img/eig_marple13_21.jpeg')

    #img_1 = import_im('./input_img/frame10_t.png')
    #img_2 = import_im('./input_img/frame11_t.png')

    img_1 = import_im('./input_img/eig_marple13_20.jpeg')
    img_2 = import_im('./input_img/eig_marple13_21.jpeg')

    frames_forward = [img_1, img_2]
    frames_backward = [img_2, img_1]

    flow_forward = flow(frames_forward)
    flow_backward = flow(frames_backward)

    flow_inverse = inverse_flow(flow_forward, flow_backward)

    tracking_point = tracking_point_matrix_threshold(flow_forward, flow_inverse)

    #vectors = np.array(list(tracking_point.values()))
    #points = np.array([list(point) for point in tracking_point.keys()])

    #img_mag = np.sqrt(flow_inverse[...,0]**2 + flow_inverse[...,1]**2)
    #plt.imshow(img_mag, cmap='gray')
    #plt.title('FarnerBack Optical Flow Magnitude')
    #plt.show()

    img_mag = magnitude(img_1.shape, tracking_point)
    plt.imshow(img_mag, cmap='gray')
    plt.show()

    img_mag = np.sqrt(flow_forward[::, ::, 0]**2 + flow_forward[::, ::, 1]**2)
    plt.imshow(img_mag, cmap='gray')
    plt.show()

    # step = 20
    # plt.quiver(points[::step,0], points[::step,1], vectors[::step,0], -1*vectors[::step,1], color='b')
    # plt.show()

    #step = 6
    #plt.imshow(img_1, cmap='gray')
    #plt.quiver(np.arange(0, flow_forward.shape[1], step), np.arange(flow_forward.shape[0], 0, -step), flow_forward[::step, ::step, 0], flow_forward[::step, ::step, 1], color='b')
    #plt.show()

    # img_mag = np.sqrt(optical_flow[...,0]**2 + optical_flow[...,1]**2)
    # plt.imshow(img_mag, cmap='gray')
    # plt.title('FarnerBack Optical Flow Magnitude')
    # plt.show()

    # flow_color = flow_vis.flow_to_color(optical_flow, convert_to_bgr=False)
    # plt.imshow(flow_color)
    # plt.title('FarnerBack Optical Flow Magnitude')
    # plt.show()
    # Use Hue, Saturation, Value colour model 

    # h, w = img_1.shape
    # hsv = np.zeros((h, w, 3))
    # hsv[:,:,1] = 255

    # mag, ang = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    # hsv[:,:,0] = ang * 180 / np.pi / 2
    # hsv[:,:,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # hsv = np.asarray(hsv, dtype= np.float32)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # cv.imshow("colored flow", bgr)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    


if __name__ == "__main__":
    main()
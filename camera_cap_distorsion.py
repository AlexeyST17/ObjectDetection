import cv2 as cv
import numpy as np
import time


# newcamera1 = np.array([[336.53872681,   0.,         366.52986482],
#  [  0.,         336.30691528, 274.68614664],
#  [  0.,           0.,           1.        ]])
# camera1 = np.array([[491.85175393,   0. ,        366.76149049],
#  [  0.,         473.72026289, 301.65891905],
#  [  0.,           0.,           1.,        ]])
# dist = np.array( [-0.31586516,  0.07931896, -0.01873824, -0.00040503,  0.00347197])



newCameraMatrix =np.array( [[2.61098281e+04, 0.00000000e+00, 6.40807345e+02],
 [0.00000000e+00, 2.60957617e+04, 4.77094235e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([-2.93334039e+00, -3.65546153e-02, -8.27704247e-03,  3.26575208e-02,
 -2.77099962e-05])
cameraMatrix = np.array([[2.62107983e+04, 0.00000000e+00, 6.39469961e+02],
 [0.00000000e+00, 2.62374599e+04, 4.77947991e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


cap = cv.VideoCapture(0)    # connect to capture device
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()     # get a frame from the capture device
    frame_counter += 1
    # print("___________-",frame)
    frame = cv.undistort(frame, cameraMatrix, dist_coeff, None, newCameraMatrix)
    # print("___________",frame)
    if ret == False:
        break
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

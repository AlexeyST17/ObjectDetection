import cv2 as cv
import time



cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()
    frame_counter += 1
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

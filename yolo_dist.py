import cv2 as cv
import time
import numpy as np
import pandas as pd
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
dist_list = []


def calc_distance(pix_width, pix_height, known_width, known_height, focal_length=621.457390525186):
    Dx = (known_width * focal_length)/pix_width
    Dy = (known_height * focal_length)/pix_height
    return (Dx + Dy)/2


class_name = []
with open('my_obj1.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('my_obj1_6000_new.weights', 'my_obj1.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        pix_height = box[3]
        pix_width = box[2]
        obj_center = (box[0]+int(pix_width/2),box[1]+int(pix_height/2))
        camera_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        camera_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        camera_center = (int(camera_width/2),int(camera_height/2))
        dist = calc_distance(pix_width=pix_width, pix_height=pix_height, known_height=17.7, known_width=15.6)
        dist_list.append(dist)
        print(f'dist: {dist} sm')
        cv.putText(frame, f'DIST: {round(dist)} sm', (20, 70),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        cv.putText(frame, 'C', obj_center, cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        cv.putText(frame, 'C1', camera_center, cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        cv.putText(frame, f'x:{camera_center[1]-obj_center[1]} y:{camera_center[0]-obj_center[0]}', (20, 400), cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.putText(frame, f'FPS: {round(fps)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


print("\nMAX: ", np.max(dist_list))
print("MIN: ", np.min(dist_list))
print("MEAN: ", np.mean(dist_list))
print("STD: ", np.std(dist_list))
print("DISPERSION: ", np.var(dist_list))

f = open('max_yolo.txt', 'a')
f.write(str(np.max(dist_list))+'\n')
f.close()

f = open('min_yolo.txt', 'a')
f.write(str(np.min(dist_list))+'\n')
f.close()

f = open('mean_yolo.txt', 'a')
f.write(str(np.mean(dist_list))+'\n')
f.close()

f = open('std_yolo.txt', 'a')
f.write(str(np.std(dist_list))+'\n')
f.close()

f = open('var_yolo.txt', 'a')
f.write(str(np.var(dist_list))+'\n')
f.close()

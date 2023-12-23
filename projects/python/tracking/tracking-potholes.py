import cv2
from ultralytics import YOLO
import os
import numpy as np
from tracker import * 
from bboxUtils import *

LIM_FRAMES = 0
LIM_IOU = 0.3

def nothing(x):
    pass

model = YOLO('Assets/models/HolesChecker_best.pt')
video_path = "Assets/videos/Pothole_Part2.mp4"
cap = cv2.VideoCapture(video_path)

tracker = DistTracker()
# name_folder = os.path.join(os.path.abspath(os.getcwd()), "Assets/results")
# if not(os.path.exists(name_folder)):
#     os.mkdir(name_folder)

curFrame = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        filter_frame = frame.copy()
        if curFrame > LIM_FRAMES:
            # Получение боксов
            result = model(frame)[0]
            boxes_data = result.boxes.data.cpu().numpy()
            boxes = np.asarray([box[:4] for box in boxes_data], dtype=int)
            confidence = [conf[4] for conf in boxes_data]

            # Отрисовка всех боксов
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Фильтрация и трекинг
            boxes = remove_small_boxes(boxes, LIM_IOU)
            boxes_ids = tracker.update(boxes)
            for box_id in boxes_ids:
                x1, y1, x2, y2, id = box_id
                cv2.rectangle(filter_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(filter_frame, (x1, y1), (x1+100, y1+50), (0, 0, 0), -1)
                cv2.putText(filter_frame, "ID: "+str(id), (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        filter_frame = cv2.resize(filter_frame, (filter_frame.shape[1]//2, filter_frame.shape[0]//2))
        cv2.imshow("All boxes", frame)
        cv2.imshow("Filter boxes", filter_frame)
        curFrame+=1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
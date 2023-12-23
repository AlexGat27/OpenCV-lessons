import cv2
from ultralytics import YOLO
import os
import numpy as np

LIM_FRAMES = 100
LIM_IOU = 0.3

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def remove_small_boxes(boxes: np.array, limit):
    if boxes.shape[0] <= 1:
        return boxes    
    for i in range(boxes.shape[0] - 1):
        for j in range(i+1, boxes.shape[0]):
            iou = intersection_over_union(boxes[i], boxes[j])
            print("IOU: " + str(iou))
            if iou > limit:
                area_i = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1)
                area_j = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1)
                
                if area_i < area_j:
                    boxes = np.delete(boxes, (i), axis=0)
                else:
                    boxes = np.delete(boxes, (j), axis=0)
    return boxes

def nothing(x):
    pass

model = YOLO('Assets/models/HolesChecker_best.pt')
video_path = "Assets/videos/Pothole_Part2.mp4"
cap = cv2.VideoCapture(video_path)

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
            print("Size all boxes: ", boxes.shape)
            for box in boxes:
                frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Фильтрация и отрисовка отфильтрованных боксов
            boxes = remove_small_boxes(boxes, LIM_IOU)
            print("Size filtered boxes: ", boxes.shape)
            for box in boxes:
                filter_frame = cv2.rectangle(filter_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

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
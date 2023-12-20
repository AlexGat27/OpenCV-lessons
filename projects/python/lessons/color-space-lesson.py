import cv2
import numpy as np

def nothing():
    pass

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hlsFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    cv2.imshow("Original Image", frame)
    cv2.imshow("RGB Image", rgbFrame)
    cv2.imshow("Gray Image", grayFrame)
    cv2.imshow("HSV Image", hsvFrame)
    cv2.imshow("HLS Image", hlsFrame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
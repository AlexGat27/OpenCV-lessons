import cv2
import numpy as np

def nothing():
    pass

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, t1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    ret, t2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
    ret, t3 = cv2.threshold(frame, 127, 255, cv2.THRESH_TRUNC)
    ret, t4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)
    ret, t5 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO_INV)
    

    cv2.imshow("Gray Image", frame)
    cv2.imshow("Binary Image", t1)
    cv2.imshow("Binary inverted Image", t2)
    cv2.imshow("Truncate Image", t3)
    cv2.imshow("Tozero Image", t4)
    cv2.imshow("Tozero inverted Image", t5)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
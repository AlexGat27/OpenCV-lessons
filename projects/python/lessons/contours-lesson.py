import cv2
import numpy as np

def nothing():
    pass

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edgeFrame = cv2.Canny(grayFrame, 100, 200)
    
    contours, h = cv2.findContours(edgeFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(frame, [contours[0]], -1, (0,0,255), 3)

    cv2.imshow("Original Image", frame)
    cv2.imshow("Gray Image", grayFrame)
    cv2.imshow("Edge Image", edgeFrame)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
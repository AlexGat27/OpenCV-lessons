import cv2
import numpy as np

def nothing(x):
    pass

kernel = np.ones((5,5))

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")

cv2.createTrackbar("HL", "frame", 0, 240, nothing)
cv2.createTrackbar("SL", "frame", 0, 255, nothing)
cv2.createTrackbar("VL", "frame", 0, 255, nothing)
cv2.createTrackbar("H", "frame", 0, 240, nothing)
cv2.createTrackbar("S", "frame", 0, 255, nothing)
cv2.createTrackbar("V", "frame", 0, 255, nothing)

while True:
    ret, frame = cap.read(0)
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("HL", "frame")
    sl = cv2.getTrackbarPos("SL", "frame")
    vl = cv2.getTrackbarPos("VL", "frame")
    h = cv2.getTrackbarPos("H", "frame")
    s = cv2.getTrackbarPos("S", "frame")
    v = cv2.getTrackbarPos("V", "frame")

    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    openingFrame = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closeFrame = cv2.morphologyEx(openingFrame, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(closeFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    rgbBounding = (0,255,0)
    for i in range(len(contours)):
        area = cv2.contourArea(contour=contours[i])
        if(area > 500):
            x,y,w,h = cv2.boundingRect(contours[i])
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), rgbBounding, 2)
            frame = cv2.rectangle(frame, (x,y), (x+60, y+25), (0,0,0), -1)
            frame = cv2.putText(frame, "My text", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    cv2.imshow("Image with detection", frame)
    cv2.imshow("Mask Image", mask)
    cv2.imshow("Filter color Image", closeFrame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#178%, 57%, 100%
#178%, 100%, 68%
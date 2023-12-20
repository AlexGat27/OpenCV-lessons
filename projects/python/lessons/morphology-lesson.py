import cv2
import numpy as np

def nothing(x):
    pass

kernel = np.ones((5,5), np.uint8)

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")
cv2.createTrackbar("HL", "frame", 0, 180, nothing)
cv2.createTrackbar("SL", "frame", 0, 255, nothing)
cv2.createTrackbar("VL", "frame", 0, 255, nothing)
cv2.createTrackbar("H", "frame", 0, 180, nothing)
cv2.createTrackbar("S", "frame", 0, 255, nothing)
cv2.createTrackbar("V", "frame", 0, 255, nothing)

while True:
    ret, frame = cap.read(0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("HL", "frame")
    sl = cv2.getTrackbarPos("SL", "frame")
    vl = cv2.getTrackbarPos("VL", "frame")
    h = cv2.getTrackbarPos("H", "frame")
    s = cv2.getTrackbarPos("S", "frame")
    v = cv2.getTrackbarPos("V", "frame")

    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])
    frame = cv2.bilateralFilter(frame, 9, 75,75)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    edgeFrame = cv2.Canny(mask, 100, 200)

    contours, h = cv2.findContours(edgeFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    try:
        cv2.drawContours(frame, [contours[0]], -1, (0,0,255), 5)
    except Exception:
        print("Ошибка")

    cv2.imshow("Bilateral Blur Image", frame)
    openingFrame = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closeFrame = cv2.morphologyEx(openingFrame, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Mask Image", mask)
    cv2.imshow("Open Close Image", closeFrame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
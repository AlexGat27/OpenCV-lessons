import cv2
import numpy as np

def nothing(x):
    pass

kernel = np.ones((3,3))

cap = cv2.VideoCapture(0)
cv2.namedWindow("Track")
cv2.createTrackbar("thresh1", "Track", 0, 255, nothing)
cv2.createTrackbar("thresh2", "Track", 0, 255, nothing)
cv2.createTrackbar("areaMaxLimit", "Track", 5000, 12000, nothing)
cv2.createTrackbar("areaMinLimit", "Track", 5000, 12000, nothing)
cv2.createTrackbar("approxPolyCoefficient", "Track", 5, 50, nothing)

while True:
    ret, frame = cap.read(0)
    t1 = cv2.getTrackbarPos("thresh1", "Track")
    t2 = cv2.getTrackbarPos("thresh2", "Track")
    areaMaxLimit = cv2.getTrackbarPos("areaMaxLimit", "Track")
    areaMinLimit = cv2.getTrackbarPos("areaMinLimit", "Track")
    approxPolyCoefficient = cv2.getTrackbarPos("approxPolyCoefficient", "Track") * 0.01
    
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edgeFrame = cv2.Canny(grayFrame, t1, t2)
    frame2contours = cv2.dilate(edgeFrame, kernel, iterations=1)
    # ret, frame2contours = cv2.threshold(grayFrame, t1, t2, cv2.THRESH_BINARY)
    
    contours, h = cv2.findContours(frame2contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area > areaMinLimit and area < areaMaxLimit):
            cv2.drawContours(frame, contour, -1, (250,210,0), 3)
            p = cv2.arcLength(contour, True)
            num = cv2.approxPolyDP(contour, approxPolyCoefficient*p, True)
            x,y,w,h = cv2.boundingRect(num)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 4)

    cv2.imshow("Original Image", frame)
    cv2.imshow("To contour Image", frame2contours)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
import cv2
import numpy as np
import os

class MatchDetector():

    def __init__(self):
        self.descTransformer = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def compareImages(self, img1, img2, _saveFolder=None):
        if type(img1)==np.str_: frame1 = cv2.imread(img1)
        else: frame1 = img1.copy()
        if type(img2)==np.str_: frame2 = cv2.imread(img2)
        else: frame2 = img2.copy()
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        keypoints1, descriptors1 = self.descTransformer.detectAndCompute(gray_frame1, None)
        keypoints2, descriptors2 = self.descTransformer.detectAndCompute(gray_frame2, None)
        matches = np.array(self.matcher.knnMatch(descriptors1, descriptors2, k=2))

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if _saveFolder != None:
            savePath = os.path.join(_saveFolder, str(len(os.listdir(_saveFolder))) + "_matchesImage.jpg")
            result = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, 
                                     None, matchColor=(0,255,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                     matchesThickness=2)
            cv2.imwrite(savePath, result)
        return len(good_matches)/len(matches)
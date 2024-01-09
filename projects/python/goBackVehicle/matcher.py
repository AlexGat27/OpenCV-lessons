import cv2
import numpy as np
import os

class MatchDetector():

    def __init__(self):
        self.descTransformer = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def compareImages(self, img1Path, img2Path, _saveFolder=None):
        frame1 = cv2.imread(img1Path)
        frame2 = cv2.imread(img2Path)
        keypoints1, descriptors1 = self.descTransformer.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = self.descTransformer.detectAndCompute(frame2, None)
        matches = np.array(self.matcher.knnMatch(descriptors1, descriptors2, k=2))

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # vectorize_matches = np.vectorize(lambda obj1, obj2: obj1.distance < 0.75 * obj2.distance)
        # goodMatches = matches[vectorize_matches(matches[:,0], matches[:,1])].tolist()
        # print(len(goodMatches))

        if _saveFolder != None:
            savePath = os.path.join(_saveFolder, str(len(os.listdir(_saveFolder))) + "_matchesImage.jpg")
            result = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, 
                                     None, matchColor=(0,255,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                     matchesThickness=2)
            cv2.imwrite(savePath, result)
        return len(good_matches)/len(matches)
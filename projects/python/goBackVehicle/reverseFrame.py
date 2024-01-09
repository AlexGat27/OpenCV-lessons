from matcher import *
import time

video_path = "Assets/videos/Pothole_Part2.mp4"
video_name = video_path.split('/')[-1].split('.')[0]
save_folder = "Assets/pictures/screenshots/validFrames/"
if os.path.exists(save_folder + video_name):
    save_folder += video_name + "_" + str(len(os.listdir(save_folder)))
else:
    save_folder += video_name
save_folder_images = save_folder + "/images"
save_coords_path = save_folder + "/imageCoords.txt"
os.makedirs(save_folder_images)

verifyImagesFolder = "Assets/pictures/screenshots/toMatchFrames/Pothole_Part2/images/"
verifyFileCoords = "Assets/pictures/screenshots/toMatchFrames/Pothole_Part2/imageCoords.txt"
verifyImagesNames = np.array(os.listdir(verifyImagesFolder))
addPathVectorize = np.vectorize(lambda name: verifyImagesFolder + name)
verifyImagesPaths = addPathVectorize(verifyImagesNames)
with open(verifyFileCoords, "r") as f:
    verifyListCoords = np.array(f.readlines())
print(verifyListCoords)

cap = cv2.VideoCapture(video_path)
timeInterval = 1
matchDetector = MatchDetector()
last_time = time.time()
currentFrameIndex = len(verifyImagesPaths) - 1

while cap.isOpened():
    success, frame = cap.read()
    if success:

        if time.time() - last_time >= timeInterval:
            matchCoef = matchDetector.compareImages(frame, verifyImagesPaths[currentFrameIndex])
            if matchCoef >= 0.9:
                filename = f'{save_folder_images}/{verifyImagesNames[currentFrameIndex]}'
                cv2.imwrite(filename, frame)
                with open(save_coords_path, 'a') as file:
                    file.write(verifyListCoords[currentFrameIndex])
                currentFrameIndex-=1
            last_time = time.time()

        cv2.imshow("Video", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        print("система не прочиатала файл")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
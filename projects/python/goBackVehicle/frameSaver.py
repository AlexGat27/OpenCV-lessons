import cv2
import os
import time
import random as rand

video_path = "Assets/videos/Pothole_Part2.mp4"
video_name = video_path.split('/')[-1].split('.')[0]
save_folder = "Assets/pictures/screenshots/toMatchFrames/"
if os.path.exists(save_folder + video_name):
    save_folder += video_name + "_" + str(len(os.listdir(save_folder)))
else:
    save_folder += video_name
save_folder_images = save_folder + "/images"
save_coords_path = save_folder + "/imageCoords.txt"
os.makedirs(save_folder_images)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = 3

last_time = time.time()
frameCount = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:

        if time.time() - last_time >= interval:
            filename = f'{save_folder_images}/frame_{frameCount}.jpg'
            cv2.imwrite(filename, frame)
            with open(save_coords_path, 'a') as file:
                file.write(f'{rand.randrange(0,50)} {rand.randrange(0, 50)}\n')
            last_time = time.time()
            frameCount+=1

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
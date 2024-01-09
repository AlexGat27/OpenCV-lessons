from matcher import *
import time

video_path = "E:/Develop/pothole-coordinates/Assets/videos/Pothole_Part2.mp4"
video_name = video_path.split('/')[-1].split('.')[0]
save_folder = "Assets/screenshots/"
if os.path.exists(save_folder + video_name):
    save_folder += video_name + "_" + str(len(os.listdir(save_folder)))
else:
    save_folder += video_name
os.mkdir(save_folder)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = 5

last_time = time.time()
frameCount = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:

        if time.time() - last_time >= interval:
            filename = f'{save_folder}/frame_{frameCount}.jpg'
            cv2.imwrite(filename, frame)
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
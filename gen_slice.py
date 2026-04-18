import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_time_space_slice(video_path, column_index, row_start, row_end):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        slice_row = frame_rgb[row_start:row_end, column_index, :]
        frames.append(slice_row)

    cap.release()

    time_slice_image = np.stack(frames, axis=0)  # shape: [frames, row_range, 3]
    time_slice_image = np.transpose(time_slice_image, (0, 1, 2))  # [T, Y, C]
    time_slice_image = np.reshape(time_slice_image, (len(frames), row_end - row_start, 3))

    return time_slice_image

video_path = 'your_video.mp4'
column_index = 100
row_start = 50
row_end = 150

slice_img = extract_time_space_slice(video_path, column_index, row_start, row_end)

plt.imshow(slice_img)
plt.title("Time-Space Slice Visualization")
plt.xlabel("Pixel Location in Row")
plt.ylabel("Time (Frame Index)")
plt.show()

import cv2

# Open the video file
cap = cv2.VideoCapture('background12.mp4')
bg_img = cv2.imread(r'E:\CHAR\python\temp\img\neg\0\1.png', cv2.IMREAD_UNCHANGED)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
# Loop over the frames of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    alpha = bg_img[:, :, 3] / 255.0

    # Paste bg_img onto frame using the alpha channel as a mask
    frame[:, :, 0] = frame[:, :, 0] * (1 - alpha) + bg_img[:, :, 0] * alpha
    frame[:, :, 1] = frame[:, :, 1] * (1 - alpha) + bg_img[:, :, 1] * alpha
    frame[:, :, 2] = frame[:, :, 2] * (1 - alpha) + bg_img[:, :, 2] * alpha

    # Write the frame to output video file
    out.write(frame)



# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

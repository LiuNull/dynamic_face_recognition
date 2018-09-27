import cv2

video_full_path = "data/video/liujunling.mp4"
cap = cv2.VideoCapture(0)
print(cap.isOpened())
frame_count = 1
success = True
while (True):
    success, frame = cap.read()
    print('Read a new frame: ', success)
    params = []
    # params.append(cv.CV_IMWRITE_PXM_BINARY)
    params.append(1)
    cv2.imwrite("video" + "_%d.jpg" % frame_count, frame, params)
    frame_count = frame_count + 1
cap.release()
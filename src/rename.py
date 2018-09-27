
import os

path = "/work/liujl/Datasets/frame_images_DB_MTCNN_NoSuffle/"

for people in os.listdir(path):
    people_path = os.path.join(path,people)
    print(people_path)
    num_video=1
    if os.path.isfile(people_path):
        continue
    for video in os.listdir(people_path):
        video_path = os.path.join(people_path, video)
        os.rename(video_path, os.path.join(people_path, "%02d" % num_video))
        num_video+=1

    for video in os.listdir(people_path):
        video_path = os.path.join(people_path,video)
        i = 1
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path,frame)
            filename = "%s_%s_%05d.png" % (people,video,i)
            i+=1
            os.rename(frame_path,os.path.join(video_path,filename))
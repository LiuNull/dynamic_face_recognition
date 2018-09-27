
# -*- coding:utf-8 -*-
import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile

# C:\Users\14542\Desktop\test


class Video():
    def __init__(self, videoname, number):
        self.videoname = videoname
        self.number = number

class People():
    def __init__(self, name, videos):
        self.name = name
        self.videos = videos


def main(args):
    total = []
    single = []
    multi = []
    peoples = os.listdir(args.input_dir)
    for people in peoples:
        if os.path.isfile(os.path.join(args.input_dir,people)):
            continue
        name = people
        if name == "Thomas_Gottschalk":
            break;
        peoplepath= os.path.join(args.input_dir,name)
        videos = os.listdir(peoplepath)
        videoslist = []
        for video in videos:
            videoname = video
            videopath = os.path.join(peoplepath,videoname)
            framenumber= len(os.listdir(videopath)) - 1
            eachvideo = Video(videoname,framenumber)
            videoslist.append(eachvideo)
        each = People(name,videoslist)

        if len(videos)==1:
            single.append(videos)
        else:
            multi.append(each)
        total.append(each)
    print('%d people have one video'% len(single))
    print('%d people have multi videos' % len(multi))

    # write the pair
    txt = open('train_pairs.txt', 'a')

    for i in range(5000):
        # write the pair of same people
        for j in range(20):
            index = np.random.randint(0,len(multi))
            videonumber = len(multi[index].videos)
            video1 = np.random.randint(0, videonumber)
            video2 = np.random.randint(0, videonumber)
            str = '%s %s %d %s %d\n' % (multi[index].name, multi[index].videos[video1].videoname,multi[index].videos[video1].number,multi[index].videos[video2].videoname,multi[index].videos[video2].number)
            txt.write(str)
    txt.close()

def arguments(argv):

    parser= argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    return parser.parse_args(argv)

    # C:\Users\14542\Desktop\t
if __name__ == '__main__':
    main(arguments(sys.argv[1:]))

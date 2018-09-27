
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

# 1595个人，其中1435个用作训练，160个人作为测试
def main(args):
    traintotal = []
    trainsingle = []
    trainmulti = []
    testtotal = []
    testsingle = []
    testmulti = []
    count=0
    peoples = os.listdir(args.input_dir)
    for people in peoples:
        if os.path.isfile(os.path.join(args.input_dir,people)):
            continue
        name = people

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
        if count<1435:
            if len(videos)==1:
                trainsingle.append(videos)
            else:
                trainmulti.append(each)
            traintotal.append(each)
        else:
            if len(videos)==1:
                testsingle.append(videos)
            else:
                testmulti.append(each)
            testtotal.append(each)
        count += 1

    print('%d trainpeople have one video'% len(trainsingle))
    print('%d trainpeople have multi videos' % len(trainmulti))
    print('%d testpeople have one video'% len(testsingle))
    print('%d testpeople have multi videos' % len(testmulti))

    # write the test_pair
    txt = open('test_pairs.txt', 'a')
    for i in range(250):
        # write the pair of same people
        for j in range(20):
            index = np.random.randint(0,len(testmulti))
            videonumber = len(testmulti[index].videos)
            video1 = np.random.randint(0, videonumber)
            video2 = np.random.randint(0, videonumber)
            str = '%s %s %d %s %d\n' % (testmulti[index].name, testmulti[index].videos[video1].videoname, testmulti[index].videos[video1].number,
                                        testmulti[index].videos[video2].videoname, testmulti[index].videos[video2].number)
            txt.write(str)
        # write the pair of different people
        for j in range(20):
            index1 = np.random.randint(0,len(testtotal))
            index2 = np.random.randint(0,len(testtotal))
            if index1==index2:
                continue
            video1= np.random.randint(0, len(testtotal[index1].videos))
            video2 = np.random.randint(0, len(testtotal[index2].videos))
            str = '%s %s %d %s %s %d\n' % (testtotal[index1].name, testtotal[index1].videos[video1].videoname, testtotal[index1].videos[video1].number, testtotal[index2].name,testtotal[index2].videos[video2].videoname, testtotal[index2].videos[video2].number)
            txt.write(str)
    txt.close()

    # write the train_pair
    txt = open('train_pairs.txt', 'a')

    for i in range(5000):
        # write the pair of same people
        for j in range(20):
            index = np.random.randint(0,len(trainmulti))
            videonumber = len(trainmulti[index].videos)
            video1 = np.random.randint(0, videonumber)
            video2 = np.random.randint(0, videonumber)
            str = '%s %s %d %s %d\n' % (trainmulti[index].name, trainmulti[index].videos[video1].videoname,trainmulti[index].videos[video1].number,trainmulti[index].videos[video2].videoname,trainmulti[index].videos[video2].number)
            txt.write(str)
    txt.close()

def arguments(argv):

    parser= argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    return parser.parse_args(argv)

    # C:\Users\14542\Desktop\t
if __name__ == '__main__':
    main(arguments(sys.argv[1:]))

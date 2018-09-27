
# -*- coding:utf-8 -*-
import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile

# C:\Users\14542\Desktop\test


class People():
    def __init__(self, name, number):
        self.name = name
        self.number = number


def main(args):
    total = []
    peoples = os.listdir(args.input_dir)
    for people in peoples:
        if os.path.isfile(os.path.join(args.input_dir,people)):
            continue
        name = people
        frames = os.listdir(os.path.join(args.input_dir,name))
        each = People(name,len(frames))
        total.append(each)
    txt = open('pairs.txt', 'a')
    for i in range(300):
        for j in range(20):
            index = np.random.randint(0,len(total))
            str = '%s %d\n' % (total[index].name, total[index].number)
            txt.write(str)
        for j in range(20):
            index1 = np.random.randint(0,len(total))
            index2 = np.random.randint(0,len(total))
            if index1==index2:
                continue
            str = '%s %d %s %d\n' %(total[index1].name, total[index1].number,total[index2].name, total[index2].number)
            txt.write(str)
    txt.close()


def arguments(argv):

    parser= argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    # parser.add_argument('file_extension', type=str, help='Directory with cropped data',default='.tif')
    return parser.parse_args(argv)

    # C:\Users\14542\Desktop\t
if __name__ == '__main__':
    main(arguments(sys.argv[1:]))

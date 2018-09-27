# -*- coding:utf-8 -*-
import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile

# C:\Users\14542\Desktop\t

def main(args):
    images_inputpath = os.path.join(args.input_dir, 'val_img')
    gtpath_inputpath = os.path.join(args.input_dir, 'val_gt')
    files = os.listdir(images_inputpath)
    txt = open('val.txt','a')
    for file in files:
        str= '%s%s%s' % (os.path.join(images_inputpath,file),' ',os.path.join(gtpath_inputpath,file))
        print(str)
        txt.write(str)
    txt.close()

'''
def main(args):
    dataset= []
    images_inputpath=os.path.join(args.input_dir,'images')
    gtpath_inputpath=os.path.join(args.input_dir,'gt')
    images_outputpath = os.path.join(args.output_dir, 'images')
    gtpath_outputpath = os.path.join(args.output_dir, 'gt')
    files = os.listdir(images_inputpath)
    for file in files:
        file_name, file_extension=os.path.splitext(file)
        img = Image.open(os.path.join(images_inputpath,file))
        gt = Image.open(os.path.join(gtpath_inputpath,file))
        for i in range(150):
            cropped_img, cropped_gt = randomCrop(img, gt)
            output_img_n = "{}_{}{}".format(file_name, i, file_extension)
            cropped_img.save(os.path.join(images_outputpath,output_img_n))
            cropped_gt.save(os.path.join(gtpath_outputpath,output_img_n))
'''

def randomCrop(image, gt):
    data_length=5000
    win_length=768
    boundary= data_length - win_length
    """
    对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
    :param image: PIL的图像image
    :return: 剪切之后的图像
    """
    crop_win_height = np.random.randint(0, boundary)
    crop_win_width= np. random.randint(0, boundary)
    random_region = (crop_win_width, crop_win_height, crop_win_width + win_length, crop_win_height+ win_length)
    return image.crop(random_region), gt.crop(random_region)

def arguments(argv):

    parser= argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str,help='Directory with raw data')
    parser.add_argument('output_dir', type=str, help='Directory with cropped data')
    # parser.add_argument('file_extension', type=str, help='Directory with cropped data',default='.tif')
    return parser.parse_args(argv)

    # C:\Users\14542\Desktop\t
if __name__ == '__main__':
    main(arguments(sys.argv[1:]))

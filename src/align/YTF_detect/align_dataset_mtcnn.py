"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
from PIL import Image,ImageDraw
import time


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class VideoClass():
    def __init__(self, name, videoset):
        self.name = name
        self.videoset = videoset


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        # 一个class代表一个people
        class_name = classes[i]
        peopledir = os.path.join(path_exp, class_name)
        videos= os.listdir(peopledir)
        videoset= []
        for video in videos:
            videopath = os.path.join(peopledir, video)
            image_paths = facenet.get_image_paths(videopath)
            video_paths = ImageClass(video, image_paths)
            videoset.append(video_paths)
        dataset.append(VideoClass(class_name, videoset))
    return dataset


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = get_dataset(args.input_dir)
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    start = time.time()

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.videoset)
            for video in cls.videoset:
                output_video_dir=os.path.join(output_class_dir, video.name)
                if not os.path.exists(output_video_dir):
                    os.makedirs(output_video_dir)

            for video in cls.videoset:
                output_video_dir = os.path.join(output_class_dir, video.name)
                for image_path in video.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_video_dir, filename+'.png')
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim<2: # 如果图像小于二维
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2: # 如果图像是二维的，将其转换成rgb格式
                                img = facenet.to_rgb(img)
                            img = img[:,:,0:3] # img是一个三维数组，其中第一二维表示长宽，第三维表示rgb

                            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces>0:
                                det = bounding_boxes[:,0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces>1:
                                    # print("There are %d faces" % nrof_faces)
                                    if args.detect_multiple_faces:
                                        temp_arr = []
                                        for i in range(nrof_faces):
                                            temp_arr.append(np.squeeze(det[i]))  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
                                        det_arr.append(temp_arr)
                                    else:
                                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                        det_arr.append(det[index,:])
                                else:
                                    det_arr.append(np.squeeze(det))
                                for i, det in enumerate(det_arr):
                                    raw_pic = Image.open(image_path)
                                    draw_pic = ImageDraw.Draw(raw_pic)
                                    if isinstance(det,list):
                                        for eachfacedet in det:
                                            eachfacedet=np.squeeze(eachfacedet)
                                            bb = np.zeros(4, dtype=np.int32)
                                            bb[0] = np.maximum(eachfacedet[0] - args.margin / 2, 0)
                                            bb[1] = np.maximum(eachfacedet[1] - args.margin / 2, 0)
                                            bb[2] = np.minimum(eachfacedet[2] + args.margin / 2, img_size[1])
                                            bb[3] = np.minimum(eachfacedet[3] + args.margin / 2, img_size[0])
                                            draw_pic.line((bb[0], bb[1], bb[2], bb[1]), "red", width=2)
                                            draw_pic.line((bb[0], bb[1], bb[0], bb[3]), "red", width=2)
                                            draw_pic.line((bb[2], bb[3], bb[2], bb[1]), "red", width=2)
                                            draw_pic.line((bb[2], bb[3], bb[0], bb[3]), "red", width=2)
                                    else:
                                        det = np.squeeze(det)
                                        bb = np.zeros(4, dtype=np.int32)
                                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                        # draw the boxing of the face
                                        # draw four lines to make a rectangle
                                        draw_pic.line((bb[0], bb[1], bb[2], bb[1]), "red", width=2)
                                        draw_pic.line((bb[0], bb[1], bb[0], bb[3]), "red", width=2)
                                        draw_pic.line((bb[2], bb[3], bb[2], bb[1]), "red", width=2)
                                        draw_pic.line((bb[2], bb[3], bb[0], bb[3]), "red", width=2)
                                        # 人脸检测的结果,人脸所在范围为cropped范围
                                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                        scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                    nrof_successfully_aligned += 1
                                    # 保存的文件名字
                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                    # raw_pic.save(output_filename_n)
                                    misc.imsave(output_filename_n, scaled)
                                    text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

    end = time.time()
    print('Total time of processing : %d, which means %g pictures per second' % (end-start, nrof_images_total/(end-start)))
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
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

import matplotlib as mpl

mpl.use('Agg')
from scipy import misc
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from matplotlib import pyplot as plt
from tensorflow.contrib import rnn
from align import align_dataset_mtcnn
import align.detect_face


def main(args):
    with tf.Graph().as_default():
        margin = args.margin
        timestep_size = 10
        raw_data_path = "/home/google/project/realtime_data"
        # raw_data_path = args.lfw_dir
        output_path = "facenet//realtime_data//mtcnn"
        threshold = 1.1
        image_size = args.image_size
        gpu_memory_fraction = 0.25

        # 建立mtcnn人脸检测模型，加载参数
        print('Creating MTCNN networks and loading parameters')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        print("successful load the MTCNN model")

        with tf.Session() as sess:
            print('Creating FaceNet networks and loading parameters')
            # sess.run(tf.global_variables_initializer())
            # Load the model
            facenet.load_model(args.model)
            print("successful load the LSTM model")
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            emb_array_LSTM_placeholder = tf.get_default_graph().get_tensor_by_name("emb_array_test_LSTM:0")
            finaloutput_placeholder = tf.get_default_graph().get_tensor_by_name("finaloutput_test:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            while True:
                peopleindex = 1
                while True:
                    path = os.path.join(raw_data_path, '%d' % peopleindex)
                    if os.path.isdir(path) == False:
                        time.sleep(0.1)
                        continue
                    # 对这个人进行人脸检测
                    images = align_data(margin,path, pnet, rnet, onet)
                    # 选取其中的timestep_size张图片进行提取融合
                    images = images[:timestep_size]
                    print("successful get images")
                    '''
                    align_dataset_mtcnn.detect("facenet//realtime_data//raw", output_path)
                    # 对检测后的图片进行重命名操作，从1开始。
                    rename(os.path.join(output_path, '%d' % peopleindex))
                    # 取出重命名后的图片路径，随机选择timestep_size张
                    paths_batch = getPath(os.path.join(outputpath, '%d' % peopleindex),timestep_size)
                    # 随机取timestep张图片进行特征计算。
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    '''
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_arr_batch = sess.run(embeddings, feed_dict=feed_dict)
                    '''
                    # 将narray类型转换为list类型
                    emb_arr_batch=emb_arr_batch.tolist()
                    # 模拟添加一个对比特征，并不做实际需要
                    emb_arr_batch.append(emb_arr_batch[0])
                    '''
                    print("trying to get the final_output")
                    finaloutput = sess.run(finaloutput_placeholder,feed_dict={emb_array_LSTM_placeholder:emb_arr_batch})
                    print("successful got the final_output")
                    compare_feature = getFeature(path)
                    print("successful got the label_feature")
                    diff = tf.subtract(finaloutput, compare_feature)
                    dist = tf.reduce_sum(tf.square(diff), axis=1)
                    print('dist: %d',dist.eval())
                    result_path=os.path.join(path,'result.txt')
                    txt = open(result_path, 'a')
                    if tf.less(dist,threshold).eval():
                        print("the same people")
                        str = "True"
                        txt.write(str)
                    else:
                        print("Not the same people")
                        str = "False"
                        txt.write(str)
                    txt.close()
                    # Success process one people
                    peopleindex += 1


def align_data(margin,image_paths, pnet, rnet, onet):
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    minsize = 20  # minimum size of face
    image_size = 160

    imagepaths=os.listdir(image_paths)
    while (True):
        count=0
        for image in imagepaths:
            count+=1
        if count>15:  #需要15张图片进行人脸检测
            break
        imagepaths = os.listdir(image_paths)
    # tmp_image_paths = image_paths.copy()
    img_list = []
    for image in imagepaths:
        print(image)
        if '.txt' in image: # 跳过feature.txt
            print(image)
            continue
        # img = misc.imread(os.path.expanduser(image), mode='RGB')
        img = misc.imread(os.path.join(image_paths, image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def getFeature(path):
    label_path = os.path.join(path,'feature.txt')
    while os.path.isfile(label_path) == False:
        time.sleep(0.2)
    # 读取txt中的数据存为numpy格式
    return np.loadtxt(label_path)



def getPath(path,timestep_size):
    paths_total=[]
    frames = os.listdir(path)
    for frame in frames:
        framepath= os.path.join(path,frame)
        paths_total.append(framepath)
    # 目前取千前10帧作为结果
    paths_timestep_size = paths_total[:timestep_size]
    return paths_timestep_size


def rename(path):
    frames = os.listdir(path)
    index = 1
    for frame in frames:
        name = index
        index += 1
        os.rename(os.path.join(path, frame), os.path.join(path, name))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=50)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
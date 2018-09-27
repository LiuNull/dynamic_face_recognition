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
import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from matplotlib import pyplot as plt
from tensorflow.contrib import rnn


def unit_lstm(hidden_size, keep_prob):
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell



def calculate_accuracy(batch_size, dist, is_same, threshold):
    predict_issame = tf.less(dist, threshold)
    # tp:预测相同，实际相同
    # fp:预测相同，实际不同
    # tn:预测不同，实际不同
    # fn:预测不同，实际相同
    tp = 0.0
    tn = 0.0
    for i in range(batch_size):
        tp_temp = tf.cond(tf.logical_and(predict_issame[i], is_same[i]), lambda: 1.0, lambda: 0.0)
        tn_temp = tf.cond(tf.logical_and(tf.logical_not(predict_issame[i]), tf.logical_not(is_same[i])), lambda: 1.0,
                          lambda: 0.0)
        tp += tp_temp
        tn += tn_temp
    acc = (tp + tn) / batch_size
    return acc, predict_issame


def get_the_best_threshold(thresholds, batch_size, dist, is_same):
    acc_train = []
    for threshold in thresholds:
        acc, _ = calculate_accuracy(batch_size, dist, is_same, threshold)
        acc_train.append(acc)
    best_threshold_index = tf.argmax(acc_train)
    return tf.gather(thresholds, best_threshold_index)


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def main(args):
    with tf.Graph().as_default():
        SUMMARY_DIR = "log/supervisor.log"
        batch_size = args.lfw_batch_size
        timestep_size = 10
        learn_rate=1e-3
        input_size = 128
        emb_array_placeholder = tf.placeholder(tf.float32, [None, input_size], name='emb_array')
        is_same_placeholder = tf.placeholder(tf.bool, [None])
        emb_arr = tf.reshape(emb_array_placeholder, [-1, (timestep_size + 1), input_size])
        # emb_frames is used to store the embeddings of the frame
        # emb_compare is used to store the embeddings of the comparing picture
        emb_frames = []
        emb_compare = []
        for i in range(batch_size):
            emb_compare.append(emb_arr[i, timestep_size, :])
            emb_frames.append(emb_arr[i, :timestep_size, :])
        emb_frames = tf.reshape(emb_frames, [-1, timestep_size, input_size])
        emb_frames = tf.reduce_mean(emb_frames,axis=1)
        diff = tf.subtract(emb_frames, emb_compare)
        dist = tf.reduce_sum(tf.square(diff), axis=1)
        thresholds = np.arange(0, 4, 0.1, dtype=np.float32)
        threshold = get_the_best_threshold(thresholds, batch_size, dist, is_same_placeholder)
        accuracy, predict_issame = calculate_accuracy(batch_size, dist, is_same_placeholder, threshold)
        # define training process
        total_loss = tf.Variable(initial_value=0, dtype=tf.float32, name='total_loss')
        # 使用正样本进行训练
        for i in range(dist.get_shape().as_list()[0]):
            loss = tf.square(dist[i])
            total_loss += loss
        total_loss /= batch_size * 10
        tf.summary.scalar('total_loss', total_loss)
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            # 考虑会不会是变量初始化导致的影响。更改变量初始化位置！
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext,timestep_size)

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = args.image_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil((1.0 * nrof_images / (timestep_size + 1)) / batch_size))
            # Run forward pass to calculate embeddings
            saver1 = tf.train.Saver()
            print('Calculate embeddings on YTF images')
            # 测试阶段
            # Test
            if i != 0:
                testpairs = lfw.read_pairs(os.path.expanduser(args.lfw_test_pairs))
                testpaths, test_actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), testpairs,
                                                              args.lfw_file_ext, timestep_size)
                test_nrof_images = len(testpaths)
                test_nrof_batches = int(math.ceil((1.0 * test_nrof_images / (timestep_size + 1)) / batch_size))
                test_dists = np.zeros(test_nrof_batches * batch_size)
                for j in range(test_nrof_batches):
                    test_start_index = j * batch_size * (timestep_size + 1)
                    test_end_index = min((j + 1) * batch_size * (timestep_size + 1), test_nrof_images)
                    test_paths_batch = testpaths[test_start_index:test_end_index]
                    test_images = facenet.load_data(test_paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: test_images, phase_train_placeholder: False}
                    test_emb_arr = sess.run(embeddings, feed_dict=feed_dict)
                    test_dis = sess.run(dist, feed_dict={emb_array_placeholder: test_emb_arr,
                                                         is_same_placeholder: test_actual_issame[
                                                                              j * batch_size:min((j + 1) * batch_size,
                                                                                                 test_nrof_images / (
                                                                                                 timestep_size + 1))]})
                    test_dists[j * batch_size:(j + 1) * batch_size] = test_dis
                thresholds = np.arange(0, 4, 0.1, dtype=np.float32)
                tpr, fpr, acc = facenet.calculate_roc_test(thresholds, test_dists, np.array(test_actual_issame))
                print('After %d iteration training, the accuracy is %.4f +- %.2f ' % (i, np.mean(acc), np.std(acc)))
                auc = metrics.auc(fpr, tpr)
                print('Area Under Curve (AUC): %1.3f' % auc)
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                print('Equal Error Rate (EER): %1.3f' % eer)
                '''
                plt.figure()
                lw = 2
                plt.figure(figsize=(10, 10))
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve')  ###假正率为横坐标，真正率为纵坐标做曲线
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                plt.savefig("roc.jpg")
                '''

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
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='train_pairs.txt')
    parser.add_argument('--lfw_test_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='test_pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
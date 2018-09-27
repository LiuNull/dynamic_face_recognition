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
        input_size = 128
        # for Train
        emb_array_placeholder = tf.placeholder(tf.float32, [None, input_size], name='emb_array_LSTM')
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
        # define LSTM
        learn_rate = 1e-3
        keep_prob = 0.5
        hidden_size = 512
        layer_num = 2
        mlstm_cell = rnn.MultiRNNCell([unit_lstm(hidden_size, keep_prob) for i in range(layer_num)],
                                      state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=emb_frames, initial_state=init_state, time_major=False)
        h_state = tf.reduce_mean(outputs[:, :, :],axis=1)
        # h_state = outputs[:, -1, :]
        cp1_W = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1, dtype=tf.float32), name='cp1_W')
        cp1_bias = tf.Variable(tf.constant(0.1, shape=[input_size], dtype=tf.float32), name='cp1_bias')
        finaloutput = tf.add(tf.matmul(h_state, cp1_W), cp1_bias, name='finaloutput')

        # for Test
        emb_array_test_placeholder = tf.placeholder(tf.float32, [None, input_size], name='emb_array_test_LSTM')
        emb_array_test_placeholder = tf.reshape(emb_array_test_placeholder, [-1, timestep_size, input_size])
        init_state_Test = mlstm_cell.zero_state(1, dtype=tf.float32)
        outputs_Test, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=emb_array_test_placeholder, initial_state=init_state_Test, time_major=False)
        h_state_test = tf.reduce_mean(outputs_Test[:, :, :], axis=1)
        finaloutput_test = tf.add(tf.matmul(h_state_test, cp1_W), cp1_bias, name='finaloutput_test')

        diff = tf.subtract(finaloutput, emb_compare)
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
        # total_loss /= batch_size * 10
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
            saver = tf.train.Saver()
            print('Calculate embeddings on YTF images')
            # 训练三十轮，每轮训练一次完整的训练数据
            for epoch in range(3):
                # 每一个epoch进行nrof_batches次迭代
                for i in range(nrof_batches):
                    start_index = i * batch_size * (timestep_size + 1)
                    end_index = min((i + 1) * batch_size * (timestep_size + 1), nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_arr_batch = sess.run(embeddings, feed_dict=feed_dict)
                    summary, los, emb_fra, emb_com, emb_agg, dis, acc, pre_is, is_sa, thre, _ = sess.run(
                        [merged, total_loss, emb_frames, emb_compare, finaloutput, dist, accuracy, predict_issame,
                         is_same_placeholder, threshold, train_op],
                        feed_dict={emb_array_placeholder: emb_arr_batch, is_same_placeholder: actual_issame[i * batch_size:min(
                                  (i + 1) * batch_size,nrof_images / (timestep_size + 1))]})
                    summary_writer.add_summary(summary, i+(epoch*nrof_batches))
                    print("The best threshold of %d epoch %d batch is %f, the loss is %f" %(epoch,i,thre,los))
                    if i % 500 ==0 and i!=0:
                        uninit_vars = []
                        for var in tf.global_variables():
                            try:
                                sess.run(var)
                            except tf.errors.FailedPreconditionError:
                                uninit_vars.append(var)
                        init_new_vars_op = tf.variables_initializer(uninit_vars)
                        sess.run(init_new_vars_op)
                        # 模型持久化
                        saver.save(sess, os.path.join('models_epoch%d_batch%d' % (epoch,i), 'LSTM.ckpt'))
                # 对未初始化的变量进行初始化



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
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

import tensorflow as tf
import argparse
import facenet
import lfw
import os
import sys
import math
from tensorflow.contrib import rnn

def unit_lstm(hidden_size,keep_prob):
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell


def calculate_accuracy(batch_size, dist, actual_issame,threshold):
    # After the training, we get the best threshold which value is 0.85
    predict_issame = tf.less(dist, threshold)
    # tp:预测相同，实际相同
    # fp:预测相同，实际不同
    # tn:预测不同，实际不同
    # fn:预测不同，实际相同
    tp=0.0
    tn=0.0
    for i in range(batch_size):
        tp_temp = tf.cond(tf.logical_and(predict_issame[i],actual_issame[i]), lambda: 1.0, lambda: 0.0)
        tn_temp = tf.cond(tf.logical_and(tf.logical_not(predict_issame[i]), tf.logical_not(actual_issame[i])), lambda: 1.0, lambda: 0.0)
        tp+=tp_temp
        tn+=tn_temp
    # tp = tf.reduce_sum(tf.logical_and(predict_issame, actual_issame))
    # fp = tf.reduce_sum(tf.logical_and(predict_issame, tf.logical_not(actual_issame)))
    # tn = tf.reduce_sum(tf.logical_and(tf.logical_not(predict_issame), tf.logical_not(actual_issame)))
    # fn = tf.reduce_sum(tf.logical_and(tf.logical_not(predict_issame), actual_issame))
    # tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    # fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc =(tp + tn) / batch_size
    return acc,predict_issame


def main(args):

    with tf.Graph().as_default():
        # define LSTM
        learn_rate = 1e-3
        batch_size = args.lfw_batch_size
        keep_prob = 0.5
        threshold = 0.85
        input_size = 128
        timestep_size = 5
        hidden_size = 128
        layer_num = 2
        mlstm_cell = rnn.MultiRNNCell([unit_lstm(hidden_size, keep_prob) for i in range(layer_num)],state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        emb_array = tf.placeholder(tf.float32,[None,input_size])
        is_same = tf.placeholder(tf.bool,[None])
        emb_arr = tf.reshape(emb_array,[-1,(timestep_size+1),input_size])
        # emb_frames is used to store the embeddings of the frame
        # emb_compare is used to store the embeddings of the comparing picture
        emb_frames = []
        emb_compare = []
        for i in range(batch_size):
            emb_compare.append(emb_arr[i,timestep_size,:])
            emb_frames.append(emb_arr[i,:timestep_size,:])
        inputs = tf.reshape(emb_frames, [-1, timestep_size, input_size])
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, initial_state=init_state, time_major=False)
        emb_aggregation_array = []
        for i in range(batch_size):
            emb_temp = outputs[i,:,:]
            emb_temp= tf.reduce_mean(emb_temp, axis=0)
            emb_temp=emb_temp/timestep_size
            emb_aggregation_array.append(emb_temp)
        # define the contrastive loss of the LSTM
        diff = tf.subtract(emb_aggregation_array, emb_compare)
        dist = tf.reduce_sum(tf.abs(diff), 1)  # the dist of two embeddings
        total_loss = tf.Variable(initial_value=0, dtype=tf.float32)

        def get_same_loss(dist,i):
            return tf.square(dist[i])

        def get_diff_loss(dist,i):
            return tf.square(tf.maximum(0.0, threshold - dist[i]))
        for i in range(dist.get_shape().as_list()[0]):
            loss=tf.cond(tf.equal(is_same[i],True), lambda : get_same_loss(dist,i), lambda : get_diff_loss(dist,i))
            total_loss+=loss
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)
        accuracy, predict_issame = calculate_accuracy(batch_size, dist, is_same, threshold)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

        # Get the paths for the corresponding images
        paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            print('Load the model')

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
        image_size = args.image_size

        # Run forward pass to calculate embeddings
        nrof_images = len(paths)
        nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
        # emb_array = np.zeros((nrof_images, embedding_size))

        print('Start Training')
        with tf.Session() as sess:
            # Train
            sess.run(tf.global_variables_initializer())
            for i in range(nrof_batches):
                start_index = i*batch_size*(timestep_size+1) # 获取batch_size * timestep_size张图片，因为每个对比都有(timestep_size张)
                end_index = min((i+1)*batch_size*(timestep_size+1), nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_arr = sess.run(embeddings, feed_dict=feed_dict)
                # until this step, everything goes right
                result,loss, acc, _=sess.run([predict_issame,total_loss,accuracy,train_op],
                                             feed_dict={emb_array:emb_arr, is_same:actual_issame[i*batch_size:(i+1)*batch_size]})
                print('After %d batches training, the accuracy is %g' % (i, acc))
                print('After %d batches training, the loss is %g' % (i, loss))
                for i in range(batch_size):
                    print(result[i])


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
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
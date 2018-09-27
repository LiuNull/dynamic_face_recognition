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


def drawpic(epoch, i, emb_agg, emb_com):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
    for index in range(len(emb_agg)):
        plt.bar(x, emb_agg[index], color='red')
        plt.savefig("pictures//%d_epoch_%d_iteration_%d_round_agg.jpg" % (epoch, i, index))
        plt.clf()
        plt.bar(x, emb_com[index], color='red')
        plt.savefig("pictures//%d_epoch_%d_iteration_%d_round_com.jpg" % (epoch, i, index))
        plt.clf()


def drawsinglepic(epoch, i, emb_fra):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
    for index in range(50):
        for round in range(10):
            plt.bar(x, emb_fra[index][round], color='red')
            plt.savefig("pictures//%d_epoch_%d_iteration_%d_round_frame_%d.jpg" % (epoch, i, index, round))
            plt.clf()


def drawtestpic(epoch, i, j,emb_agg, emb_com):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
    for index in range(len(emb_agg)):
        plt.bar(x, emb_agg[index], color='red')
        plt.savefig("pictures//%d_epoch_%d_iteration_%d_testBatch_%d_round_agg.jpg" % (epoch, i, j,index))
        plt.clf()
        plt.bar(x, emb_com[index], color='red')
        plt.savefig("pictures//%d_epoch_%d_iteration_%d_testBatch_%d_round_com.jpg" % (epoch, i, j,index))
        plt.clf()


def drawtestsinglepic(epoch, i,j, emb_fra):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
         76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
    for index in range(50):
        for round in range(10):
            plt.bar(x, emb_fra[index][round], color='red')
            plt.savefig("testpictures//%d_epoch_%d_iteration_%d_testBatch_%d_round_frame_%d.jpg" % (epoch, i, j,index, round))
            plt.clf()



def calculate_accuracy(batch_size, dist, is_same, threshold):
    predict_issame = tf.less(dist, threshold)  # 判断哪些欧式距离是低于阈值的，预测他们是同一个人
    # tp:预测相同，实际相同
    # fp:预测相同，实际不同
    # tn:预测不同，实际不同
    # fn:预测不同，实际相同
    tp = 0.0
    fn = 0.0
    for i in range(batch_size):
        # 因为predict_issame和is_same是tensor，如果用if判断会报错，所以用tf封装的判断方法进行判断，若为true则返回1，否则返回0
        tp_temp = tf.cond(tf.logical_and(predict_issame[i], is_same[i]), lambda: 1.0, lambda: 0.0)
        fn_temp = tf.cond(tf.logical_and(tf.logical_not(predict_issame[i]), tf.logical_not(is_same[i])), lambda: 1.0,
                          lambda: 0.0)
        tp += tp_temp
        fn += fn_temp
    acc = (tp + fn) / batch_size  # 计算准确率
    return acc, predict_issame


def get_the_best_threshold(thresholds, batch_size, dist, is_same):
    acc_train = []  # 用来保存每个阈值下计算得到的准确率
    for threshold in thresholds:
        acc, _ = calculate_accuracy(batch_size, dist, is_same, threshold)  # 计算准确率
        acc_train.append(acc)
    best_threshold_index = tf.argmax(acc_train)  # 获取最大准确率时的索引
    return tf.gather(thresholds, best_threshold_index)  # 获取thresholds中索引为best_threshold_index的具体值


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def main(args):
    with tf.Graph().as_default():
        SUMMARY_DIR = "log/supervisor.log"  # 定义tensorboard 文件保存的路径
        batch_size = args.lfw_batch_size   # 定义batch_size的大小，arg参数在代码最底下定义了
        timestep_size = 20   # timestep_size长度即为LSTM输入的长度，定为10.表示一组输入有10个时序特征，每个特征128维
        input_size = 128  # input_size为128，说明输入的时序特征长度为128
        emb_array_placeholder = tf.placeholder(tf.float32, [None, input_size], name='emb_array')  # 定义LSTM输入数据的入口。输入数据包含训练数据和标签label
        is_same_placeholder = tf.placeholder(tf.bool, [None])   # 用于计算准确率，是否真正是同一个人，和模型计算得到的结果进行比较。
        emb_arr = tf.reshape(emb_array_placeholder, [-1, (timestep_size + 1), input_size])   # 对LSTM输入数据进行转换，是其输入数据拆分成[50,10,128]的训练数据和[50,1,128]的标签数据
        emb_frames = []  # emb_frames is used to store the embeddings of the frame
        emb_compare = []  # emb_compare is used to store the embeddings of the comparing picture
        for i in range(batch_size):
            emb_compare.append(emb_arr[i, timestep_size, :])  # 用来保存标签数据
            emb_frames.append(emb_arr[i, :timestep_size, :])  # 用来保存训练数据
        emb_frames = tf.reshape(emb_frames, [-1, timestep_size, input_size])
        # define LSTM
        learn_rate = 1e-3  # 定义学习率
        keep_prob = 0.5  # 我去整理一下这个知识点再来解释
        hidden_size = 512  # LSTM中网络的隐含层结点数
        layer_num = 2   # LSTM层数，有两层，第一层的输出为第二层的输入。最后获取第二层的输出进行平均
        # 定义LSTM的结构体，unit_lstm的函数定义在代码最上面。这段代码的意思是定义了一个两层的结构体
        mlstm_cell = rnn.MultiRNNCell([unit_lstm(hidden_size, keep_prob) for i in range(layer_num)],
                                      state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化LSTM的状态，初始化为全0
        outputs, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=emb_frames, initial_state=init_state, time_major=False)  # 进行LSTM的计算，最终得到一组10个特征的输出，每个特征的大小等于隐含层结点数512.具体LSTM如何计算可以参考我的论文中的LSTM三个门结构的介绍
        h_state = tf.reduce_mean(outputs[:, :, :],axis=1)  # 得到的是一组10个特征的输出，将其进行简单平均，以得到一个平均的512维的特征
        # h_state = outputs[:, -1, :]
        # 为了将LSTM得到的512维特征和正脸的128维特征进行计算，在LSTM网络后添加一个全连接层，结点数为128，以将512维的特征映射回128维FaceNet特征空间
        cp1_W = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1, dtype=tf.float32), name='cp1_W')  # 定义全连接层的权重
        cp1_bias = tf.Variable(tf.constant(0.1, shape=[input_size], dtype=tf.float32), name='cp1_bias')  # 定义全连接层的偏置

        finaloutput = tf.add(tf.matmul(h_state, cp1_W), cp1_bias, name='finaloutput')  # 通过全连接层的计算得到最终的128维的输出

        diff = tf.subtract(finaloutput, emb_compare)
        dist = tf.reduce_sum(tf.square(diff), axis=1)  # 计算最终输出与label之间的欧氏距离
        # threshold是阈值，若输出和label的欧式距离高于阈值，则认为不是同一个人，否则认为是同一个人。
        # 阈值的选取是通过遍历得到，即从0开始，每隔0.1取一次值，然后分别对同一个训练数据进行准确率计算，准确率最高的值作为选定的阈值
        thresholds = np.arange(0, 4, 0.1, dtype=np.float32)
        threshold = get_the_best_threshold(thresholds, batch_size, dist, is_same_placeholder)  # 获取准确率最大对应的阈值
        accuracy, predict_issame = calculate_accuracy(batch_size, dist, is_same_placeholder, threshold)   # 计算准确率
        # define training process
        total_loss = tf.Variable(initial_value=0, dtype=tf.float32, name='total_loss')
        # 使用正样本进行训练
        for i in range(dist.get_shape().as_list()[0]):
            loss = tf.square(dist[i])  # 定义损失函数为欧式距离
            total_loss += loss  # 将整个batch的欧氏距离作为最终的损失函数
        total_loss /= batch_size * 10  # 这个步骤是多余的，是为了让损失值看上去小一点，哈哈哈哈
        tf.summary.scalar('total_loss', total_loss)  # 将损失值添加至tensorboard文件中
        # 定义训练过程，使用adam优化器，使得loss值越小越好
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            # 考虑会不会是变量初始化导致的影响。更改变量初始化位置！
            sess.run(tf.global_variables_initializer())  # 初始化参数
            summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))  # 获取训练数据，这是一个txt文件，可以进去看一下具体的函数。
            # TXT文件示例如下：
            # Franklin_Brown Franklin_Brown_01 157 Franklin_Brown_03 48
            # 训练数据全是正样本，即全是相同人的比较。也可以看作为了训练网络使得一个人的视频提取的特征与正脸的特征相似
            # 其中第一个字段表示人名，第二个字段表示选择这个人的哪个视频段，第三个字段表示该视频段的帧数。
            # 其中第四个字段该人的第二个视频段，第五个字段表示第二个视频段的帧数。

            # Get the paths for the corresponding images
            # 根据人名获取对应文件夹中的视频帧进行处理
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext,timestep_size)

            # Load the model
            facenet.load_model(args.model)  # 加载FaceNet模型

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # 定义FaceNet的训练数据的入口
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # 定义获取FaceNet计算结果的出口
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")  # 这个我也不知道

            image_size = args.image_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil((1.0 * nrof_images / (timestep_size + 1)) / batch_size))
            # Run forward pass to calculate embeddings
            saver1 = tf.train.Saver()
            print('Calculate embeddings on YTF images')
            # 训练三轮，每轮训练一次完整的训练数据
            for epoch in range(3):  # 训练三个epoch，每一个epoch有50000个训练数据，每一个batch有50个训练数据，也就是一个epoch有1000个batch。训练数据是视-正脸频图像对，共有1000个人，每个人可以从0s开始，从1s开始进行时序截取10帧。所以每个人可以有多个训练数据。
                # 每一个epoch进行nrof_batches次迭代
                for i in range(nrof_batches):
                    start_index = i * batch_size * (timestep_size + 1)  # 第i个batch对应的训练数据
                    end_index = min((i + 1) * batch_size * (timestep_size + 1), nrof_images)
                    paths_batch = paths[start_index:end_index]  # 获取第i个batch对应的50个训练数据的路径
                    images = facenet.load_data(paths_batch, False, False, image_size)  # 通过路径获取图片
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}  # 向定义的输入处往FaceNet里面喂数据
                    emb_arr_batch = sess.run(embeddings, feed_dict=feed_dict)  # 从定义输出处获取计算得到的特征

                    # 向LSTM定义阶段定义的LSTM网络数据入口喂两个数据，第一个是FaceNet计算得到的特征，第二个是实际的是否为同一个人的结果用以计算准确率。
                    summary, _= sess.run(
                        [merged, train_op],feed_dict={emb_array_placeholder: emb_arr_batch, is_same_placeholder: actual_issame[i * batch_size:min(
                                  (i + 1) * batch_size,nrof_images / (timestep_size + 1))]})
                    summary_writer.add_summary(summary, i+(epoch*nrof_batches))
                    '''
                    # 保存训练结果以及绘制特征
                    txt=open('loss.txt','a')
                    txt.write('%f\n' % los)
                    txt.close()

                    if i % 500 == 0 and i != 0:
                        drawpic(epoch, i, emb_com, emb_agg)
                        drawsinglepic(epoch, i, emb_fra)
                    '''
                    '''
                    # Test
                    if i % 100 == 0 and i != 0:
                        testpairs = lfw.read_pairs(os.path.expanduser(args.lfw_test_pairs))
                        testpaths, test_actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), testpairs,
                                                                      args.lfw_file_ext, timestep_size)
                        test_images = facenet.load_data(testpaths, False, False, image_size)
                        feed_dict = {images_placeholder: test_images, phase_train_placeholder: False}
                        test_emb_arr = sess.run(embeddings, feed_dict=feed_dict)
                        test_acc, test_thre, test_pre_is, test_is_sa, test_dis = sess.run(
                            [accuracy, threshold,predict_issame,is_same_placeholder,dist],
                            feed_dict={emb_array_placeholder: test_emb_arr, is_same_placeholder: test_actual_issame})
                        print('After %d epoch training, the accuracy is %.4f and the threshold is %.1f' % (
                            epoch, test_acc, test_thre))
                        txt = open('debug.txt', 'a')
                        for k in range(100):
                            str = 'After %d epoch %d iteration training, in %d round,the predict is %s, the actual is %s, and the dist is %f \n  ' % (
                            epoch, i,  k, test_pre_is[k], test_is_sa[k], test_dis[k])
                            txt.write(str)
                            str = 'the picture of the compare is %s \n' % \
                                  testpaths[timestep_size::(1 + timestep_size)][k]  # 从10开始，步长为timestep_size+1
                            txt.write(str)
                        txt.close()
                        drawtestpic(epoch, i, emb_com, emb_agg)
                        drawtestsinglepic(epoch, i, emb_fra)
                    '''

                    # 测试阶段
                    # Test
                    if i % 100 == 0 and i != 0: # 每训练100轮，进行一次测试
                        # 获取测试数据
                        # 测试数据格式如下：
                        # James_Kirtley James_Kirtley_02 200 James_Kirtley_02 200
                        # Idi_Amin Idi_Amin_02 56 Francisco_Garcia Francisco_Garcia_02 79
                        # 其中有两种类型的测试数据，分别是相同人的测试数据和不同人的测试数据。
                        # 相同人的测试数据就和训练数据一样。
                        # 不同人的测试数据解释：其中第一个字段表示第一个人名，第二个字段表示该人的某个视频段，第三个字段表示该视频段的帧数
                        # 第四个字段表示第二个人名，第五个字段表示第二个人的某个视频段，第六个字段表示该视频段的帧数
                        testpairs = lfw.read_pairs(os.path.expanduser(args.lfw_test_pairs))
                        # 获取对应测试数据图片的路径
                        testpaths, test_actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), testpairs,
                                                                      args.lfw_file_ext,timestep_size)
                        test_nrof_images = len(testpaths)
                        test_nrof_batches = int(math.ceil((1.0 * test_nrof_images / (timestep_size + 1)) / batch_size))
                        test_dists = np.zeros(test_nrof_batches* batch_size)
                        for j in range(test_nrof_batches):
                            start = time.time()
                            # 第j个batch对应的测试数据的索引
                            test_start_index = j * batch_size * (timestep_size + 1)
                            test_end_index = min((j + 1) * batch_size * (timestep_size + 1), test_nrof_images)
                            test_paths_batch = testpaths[test_start_index:test_end_index]
                            # 根据路径获取图片
                            test_images = facenet.load_data(test_paths_batch, False, False, image_size)
                            # 喂数据至FaceNet
                            feed_dict = {images_placeholder: test_images, phase_train_placeholder: False}
                            # 获取FaceNet计算得到的特征组
                            test_emb_arr = sess.run(embeddings, feed_dict=feed_dict)
                            # 将该数据传入LSTM经过计算得到欧氏距离
                            test_dis= sess.run(dist,feed_dict={emb_array_placeholder: test_emb_arr,
                             is_same_placeholder: test_actual_issame[j * batch_size:min((j + 1) * batch_size,test_nrof_images / (timestep_size + 1))]})
                            '''
                            end = time.time()
                            txt = open('time.txt','a')
                            str = 'After %d batch training, the time of test_batch %d is %d \n' % (i,j,end-start)
                            txt.write(str)
                            txt.close()
                            if i %300==0 and j==0:
                                drawtestpic(epoch, i, j, emb_com, emb_agg)
                                drawtestsinglepic(epoch, i, j, emb_fra)
                            txt= open('debug.txt','a')
                            for k in range(batch_size):
                                str = 'After %d epoch %d iteration training, in %d test_batch %d round,the predict is %s, the actual is %s, and the dist is %f \n  ' %(epoch, i,j,k,test_pre_is[k],test_is_sa[k],test_dis[k])
                                txt.write(str)
                                str = 'the picture of the compare is %s \n' % test_paths_batch[timestep_size::(1+timestep_size)][k] #从10开始，步长为timestep_size+1
                                txt.write(str)
                            txt.close()
                              '''
                            test_dists[j * batch_size:(j + 1) * batch_size]=test_dis
                        thresholds = np.arange(0, 4, 0.1, dtype=np.float32)
                        # 计算准确率，这个facenet.calculate_roc_test方法可以具体看一下。acc是准确率，fpr和tpr是用来计算AUC值的，AUC值的具体含义可以看我的论文中评价指标一小节。
                        tpr, fpr, acc = facenet.calculate_roc_test(thresholds,test_dists,np.array(test_actual_issame))
                        print('After %d epoch %d iteration training, the accuracy is %.4f +- %.2f and the threshold is %.1f' % (epoch, i, np.mean(acc), np.std(acc)))
                        auc = metrics.auc(fpr, tpr)
                        print('Area Under Curve (AUC): %1.3f' % auc)


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
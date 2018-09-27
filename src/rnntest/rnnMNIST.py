import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


sess=tf.Session()

mnist=input_data.read_data_sets('../../../data/MNIST_data',one_hot=True)

lr=1e-3
batch_size=tf.placeholder(tf.int32,[])
keep_prob=tf.placeholder(tf.float32)

input_size=128
timestep_size=5
hidden_size=256
layer_num=2
class_num=2 # whether it is the same people or not

_x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,class_num])

# input shape= [batch_size, timestep_size, inputs]
x=tf.reshape(_x,[-1,timestep_size,input_size])
#  num_units 在LSTM cell中unit的数目， forget_bias 添加到遗忘门中的偏置， input_size 输入到LSTM cell中的输入维度，默认等于num_units
#state_is_tuple 表示状态c和h分开记录，放入一个tuple中，如果参数没有设定或设置为False，两个状态就按列连接起来，成为[batch_size, 2*hidden_units]
#lstm_cell=rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0,state_is_tuple=True)
# input_keep_prob=1 means no input dropout will be added
# lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)


def unit_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell


mlstm_cell = rnn.MultiRNNCell([unit_lstm() for i in range(layer_num)],state_is_tuple=True)
# mlstm_cell=unit_lstm()
init_state= mlstm_cell.zero_state(batch_size,dtype=tf.float32)
# if inputs=(batches,steps,inputs) -->time_step=False
# if inputs=(steps,batches,inputs) -->time_step=True
outputs, state=tf.nn.dynamic_rnn(mlstm_cell,inputs=x,initial_state=init_state,time_major=False)
h_state=outputs[:,-1,:]
# 定义output集合用于存放各个时刻的输出，output集合中的数据已经包含了时序信息，对output集合中的特征进行融合即可。
'''
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = mlstm_cell(x[:, timestep, :],state)
        outputs.append(cell_output)
h_state = outputs[-1]
'''
# 定义一个全连接层，用于分类处理
W=tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1,dtype=tf.float32))
bias=tf.Variable(tf.constant(0.1,shape=[class_num],dtype=tf.float32))
y_pre = tf.nn.softmax(tf.matmul(h_state,W)+bias)
cross_entropy= -tf.reduce_mean(y*tf.log(y_pre))
train_op=tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# argmax 中的 1 代表axis=1，即对列进行比较
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(y_pre,1))
accurary=tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size=128
    batch=mnist.train.next_batch(_batch_size)
    if (i+1)%200 ==0 :
        train_accuracy = sess.run(accurary,feed_dict={_x:batch[0],y:batch[1],keep_prob:1.0,batch_size:_batch_size})
        print("Iter%d, step%d,training accuracy %g" % (mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op,feed_dict={_x:batch[0],y:batch[1],keep_prob:0.5,batch_size:_batch_size})

print("test accuracy %g" % sess.run(accurary,feed_dict={_x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0, batch_size:mnist.test.images.shape[0]}))

import tensorflow as tf
import os.path
input_holder = tf.placeholder(tf.float32,shape=[1],name="input_holder")
W1 = tf.Variable(tf.constant(5.0, shape=[1]), name="W1")
B1 = tf.Variable(tf.constant(1.0, shape=[1]), name="B1")
_y = (input_holder * W1) + B1
predictions = tf.greater(_y, 50, name="predictions")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    predictions=sess.run(predictions, feed_dict={input_holder: [10.0]})
    print("predictions : ", predictions)
    saver = tf.train.Saver()
    saver.save(sess,os.path.join('models','LSTM.ckpt'))
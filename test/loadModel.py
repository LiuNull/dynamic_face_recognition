import tensorflow as tf
import os.path
from tensorflow.python.platform import gfile

def main():

    a=tf.placeholder(tf.bool,None,name='resultPlaceHoler')
    b=tf.Variable(dtype=tf.float32,initial_value=2.0)
    if a is True:
        b+=tf.Variable(dtype=tf.float32,initial_value=1.0)
    else:
        b+=tf.Variable(dtype=tf.float32,initial_value=2.0)
    c=tf.add(b,tf.Variable(dtype=tf.float32,initial_value=0),name='output')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

    with tf.Session() as sess:
        load_model('models')

    input_holder = tf.get_default_graph().get_tensor_by_name("input_holder:0")
    predictions = tf.get_default_graph().get_tensor_by_name("predictions:0")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predi = sess.run(predictions,feed_dict={input_holder:[10.0]})
        qufan = sess.run(b,feed_dict={a:predi})
        print(qufan)
        saver.save(sess, os.path.join('models_2', 'LSTM.ckpt'))

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


if __name__ == '__main__':
    main()
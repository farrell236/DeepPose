# ================================================================================
# Copyright (c) 2018 Benjamin Hou (bh1511@imperial.ac.uk)
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
# ================================================================================

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
from tqdm import tqdm

# ================================================================================
# Reader Class
# ================================================================================

class PoseNetReader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(tfrecord_list)

    def read_and_decode(self):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.file_q)

        features = tf.parse_single_example(
            serialized_example,
            features={
                #'height':      tf.FixedLenFeature([], tf.int64),
                #'width':       tf.FixedLenFeature([], tf.int64),
                'image':        tf.FixedLenFeature([], tf.string),
                'pose_q':       tf.FixedLenFeature([], tf.string),
                'pose_x':       tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        pose_q = tf.decode_raw(features['pose_q'], tf.float32)
        pose_x = tf.decode_raw(features['pose_x'], tf.float32)

        #height = tf.cast(features['height'], tf.int32)
        #width = tf.cast(features['width'], tf.int32)

        image = tf.reshape(image, (480, 270, 3))
        pose_q.set_shape((4))
        pose_x.set_shape((3))

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        #image = tf.image.resize_images(image, size=[224, 224])

        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=224,
                                                       target_width=224)

        image_batch , pose_q_batch , pose_x_batch = tf.train.shuffle_batch([image, pose_q, pose_x],
                                                                           batch_size=64,
                                                                           capacity=1024,
                                                                           num_threads=2,
                                                                           min_after_dequeue=10)

        #return image, pose_q, pose_x
        return image_batch , pose_q_batch , pose_x_batch

# ================================================================================
# Network Definition
# ================================================================================

logs_path = 'logs/'
ckpt_path = 'model_ckpt/model_'

# create a list of all our filenames
filename_train = ['../dataset/KingsCollege/dataset_train.tfrecords']
#filename_test = ['../dataset/KingsCollege/dataset_test.tfrecords']

reader_train = PoseNetReader(filename_train)
#reader_eval = PoseNetReader(filename_test)

# Get Input Tensors
image, pose_q, pose_x = reader_train.read_and_decode()
#image, pose_q, pose_x = reader_eval.read_and_decode()


# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    py_x , _ = inception.inception_v3(tf.cast(image,tf.float32))

    py_x = tf.nn.relu(py_x)

    weights = {
        'h1': tf.Variable(tf.random_normal([1000, 4]),name='w_wpqr_out'),
        'h2': tf.Variable(tf.random_normal([1000, 3]),name='w_xyz_out'),
    }
    biases = {
        'b1': tf.Variable(tf.zeros([4]),name='b_wpqr_out'),
        'b2': tf.Variable(tf.zeros([3]),name='b_xyz_out'),
    }

    cls3_fc_pose_wpqr = tf.add(tf.matmul(py_x, weights['h1']), biases['b1'])
    cls3_fc_pose_xyz = tf.add(tf.matmul(py_x, weights['h2']), biases['b2'])

with tf.name_scope('Loss'):
    # Minimize error using weighted Euclidean Distance
    cost_wpqr = tf.reduce_mean(tf.squared_difference(cls3_fc_pose_wpqr, pose_q))
    cost_xyz = tf.reduce_mean(tf.squared_difference(cls3_fc_pose_xyz, pose_x))

    cost = tf.add(500*cost_wpqr,cost_xyz)

with tf.name_scope('Adam'):
    # Adam Optimiser
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
    #predict_op = tf.argmax(py_x, 1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar('loss', cost)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# ================================================================================
# Training Runtime
# ================================================================================

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Start Queue Threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    saver = tf.train.Saver()
    #latest_checkpoint = tf.train.latest_checkpoint('model_ckpt')
    #saver.restore(sess, latest_checkpoint)

    # Training cycle
    try:
        for i in tqdm(range(200000)):

            _, _cost, summary = sess.run([train_op, cost, merged_summary_op])

            # Write logs at every iteration
            print 'iteration: ', i, ' cost: ', _cost
            summary_writer.add_summary(summary, i)

            if i % 10000 == 0:
                save_path = saver.save(sess, ckpt_path + str(i) + '.ckpt')

    except KeyboardInterrupt:
        print 'KeyboardInterrupt!'

    finally:
        print 'Stopping Threads'
        coord.request_stop()
        coord.join(threads)
        print 'Saving iter: ', i
        save_path = saver.save(sess, ckpt_path + str(i) + '.ckpt')







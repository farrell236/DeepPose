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
import math
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception

# ================================================================================
# Reader Class
# ================================================================================

class PoseNetReader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(tfrecord_list,num_epochs=1)

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

        image = tf.reshape(image, (1, 480, 270, 3))
        pose_q.set_shape((4))
        pose_x.set_shape((3))

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        #image = tf.image.resize_images(image, size=[224, 224])

        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=224,
                                                       target_width=224)

        return image , pose_q , pose_x

# ================================================================================
# Network Definition
# ================================================================================

logs_path = 'logs/'
ckpt_path = 'model_ckpt'

# create a list of all our filenames
#filename_train = ['dataset/KingsCollege/dataset_train.tfrecords']
filename_test = ['../dataset/KingsCollege/dataset_test.tfrecords']

#reader_train = PoseNetReader(filename_train)
reader_eval = PoseNetReader(filename_test)

# Get Input Tensors
#image, pose_q, pose_x = reader_train.read_and_decode()
image, pose_q, pose_x = reader_eval.read_and_decode()


# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    py_x , _ = inception.inception_v3(tf.cast(image,tf.float32),is_training=False)

    py_x = tf.nn.relu(py_x)

    weights = {
        'h1': tf.Variable(tf.random_normal([1000, 4]),name='w_wpqr_out'),
        'h2': tf.Variable(tf.random_normal([1000, 3]),name='w_xyz_out'),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([4]),name='b_wpqr_out'),
        'b2': tf.Variable(tf.random_normal([3]),name='b_xyz_out'),
    }

    cls3_fc_pose_wpqr = tf.add(tf.matmul(py_x, weights['h1']), biases['b1'])
    cls3_fc_pose_xyz = tf.add(tf.matmul(py_x, weights['h2']), biases['b2'])

# ================================================================================
# Evaluation Runtime
# ================================================================================

# Create Tensorflow Session
sess = tf.Session()
sess.run(tf.initialize_local_variables())

# Start Queue Threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)

# Restore Model and Weights
saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint(ckpt_path)
saver.restore(sess,latest_checkpoint)

# Evaluate Network
results = np.zeros((400,2))

try:

    for i in range(0,400):

        _cls3_fc_pose_wpqr , _cls3_fc_pose_xyz , _pose_q , _pose_x = \
            sess.run([cls3_fc_pose_wpqr , cls3_fc_pose_xyz , pose_q , pose_x])

        #print _cls3_fc_pose_wpqr , _pose_q
        #print _cls3_fc_pose_xyz , _pose_x

        _pose_q = np.squeeze(_pose_q)
        _pose_x = np.squeeze(_pose_x)
        _cls3_fc_pose_wpqr = np.squeeze(_cls3_fc_pose_wpqr)
        _cls3_fc_pose_xyz = np.squeeze(_cls3_fc_pose_xyz)

        #Compute Individual Sample Error
        q1 = _pose_q / np.linalg.norm(_pose_q)
        q2 = _cls3_fc_pose_wpqr / np.linalg.norm(_cls3_fc_pose_wpqr)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(_pose_x-_cls3_fc_pose_xyz)

        results[i,:] = [error_x,theta]

        print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

except tf.errors.OutOfRangeError:
    print('End of Test Data')

finally:
    coord.request_stop()
    coord.join(threads)

median_result = np.median(results,axis=0)
mean_result = np.mean(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'
print 'Mean error ', mean_result[0], 'm  and ', mean_result[1], 'degrees.'

np.savetxt('results.txt', results, delimiter=' ')

print 'Success!'



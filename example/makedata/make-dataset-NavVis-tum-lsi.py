import os
import xml.etree.cElementTree as ET
from random import shuffle

import imageio
import numpy as np
import tensorflow as tf
from skimage import exposure
from tqdm import tqdm

import sys
sys.path.append('/vol/medic01/users/bh1511/_build/geomstats-master/')
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


###############################################################################
# Tensorflow feature wrapper

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.floatlist(value=[value]))


###############################################################################
# Dataset Locations

TUM_LSI_IMAGES = 'NavVis-Indoor-Dataset/images/'
TUM_LSI_LABELS = 'NavVis-Indoor-Dataset/poses/2015-08-16_15.34.11_poses.xml'

TUM_LSI_FILE_LIST = 'NavVis-Indoor-Dataset/tum-lsi_train.txt'

TF_RECORD_DATABASE = 'dataset_train.tfrecords'


###############################################################################
# Process Lables

with open(TUM_LSI_FILE_LIST) as f:
    content = f.readlines()

content = [x.strip() for x in content]

shuffle(content)
shuffle(content)
shuffle(content)

xml = ET.parse(TUM_LSI_LABELS).getroot()
poses = {}

for i in tqdm(range(0, len(xml))):
    poses[xml[i][0].text] = {}
    poses[xml[i][0].text]['x'] = float(xml[i][1][0].text)
    poses[xml[i][0].text]['y'] = float(xml[i][1][1].text)
    poses[xml[i][0].text]['z'] = float(xml[i][1][2].text)
    poses[xml[i][0].text]['w'] = float(xml[i][2][0].text)
    poses[xml[i][0].text]['p'] = float(xml[i][2][1].text)
    poses[xml[i][0].text]['q'] = float(xml[i][2][2].text)
    poses[xml[i][0].text]['r'] = float(xml[i][2][3].text)


###############################################################################
# Create Database

SO3_GROUP = SpecialOrthogonalGroup(3)
writer = tf.python_io.TFRecordWriter(TF_RECORD_DATABASE)

for i in tqdm(range(0, len(content))):
    fname = os.path.basename(content[i]).replace('.jpg', '')

    x = poses[fname]['x']
    y = poses[fname]['y']
    z = poses[fname]['z']
    w = poses[fname]['w']
    p = poses[fname]['p']
    q = poses[fname]['q']
    r = poses[fname]['r']

    pose_q = np.array([w, p, q, r])
    pose_x = np.array([x, y, z])

    rot_vec = SO3_GROUP.rotation_vector_from_quaternion(pose_q)[0]
    pose    = np.concatenate((rot_vec, pose_x), axis=0)

    X = imageio.imread(TUM_LSI_IMAGES+content[i])
    X = X[::8, ::8]
    X = exposure.equalize_hist(X)

    img_raw     = X.astype('float32').tostring()
    pose_raw    = pose.astype('float32').tostring()
    pose_q_raw  = pose_q.astype('float32').tostring()
    pose_x_raw  = pose_x.astype('float32').tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height':   _int64_feature(X.shape[0]),
        'width':    _int64_feature(X.shape[1]),
        'channel':  _int64_feature(X.shape[2]),
        'image':    _bytes_feature(img_raw),
        'pose':     _bytes_feature(pose_raw),
        'pose_q':   _bytes_feature(pose_q_raw),
        'pose_x':   _bytes_feature(pose_x_raw)}))

    writer.write(example.SerializeToString())

    # imageio.imsave('proc/'+fname+'.png', X)

writer.close()

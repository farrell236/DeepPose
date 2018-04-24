"""
Example usage of se3_geodesic_loss for TensorFlow Framework

Input tensors:
    y_pred : predicted se3 pose (shape = [Nx6])
    y_true : ground truth se3 pose (shape = [Nx6])

Output tensor:
    loss   : total squared geodesic loss of all N examples

N.B. Gradient tensor (shape = [Nx6]) is only calculated w.r.t. y_pred.
Gradient w.r.t. y_true is a vector of ones.
"""

import numpy as np
import tensorflow as tf
from se3_geodesic_loss import SE3GeodesicLoss


N_SAMPLES = 20
SE3_DIM = 6

loss = SE3GeodesicLoss(np.ones(SE3_DIM))

with tf.Session() as sess:

    # Create random input variables
    # y_pred = tf.random_normal([N_SAMPLES, SE3_DIM])
    # y_true = tf.random_normal([N_SAMPLES, SE3_DIM])

    # Create deterministic input variables
    y_pred = tf.constant([1., 2., 3., 4., 5., 6.], shape=[N_SAMPLES, SE3_DIM])
    y_true = tf.constant([3., 4., 5., 6., 7., 8.], shape=[N_SAMPLES, SE3_DIM])

    # Create graph
    geodesic_loss = loss.geodesic_loss(y_pred, y_true)

    # Compute forward loss
    print sess.run(geodesic_loss)

    # Compute backward gradients
    print sess.run(tf.gradients(geodesic_loss, [y_pred, y_true]))

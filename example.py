from se3_geodesic_loss import SE3GeodesicLoss
import tensorflow as tf
import numpy as np

## init SE3GeodesicLoss class with inner_product_mat_at_identity weights
loss = SE3GeodesicLoss(np.array([1,1,1,1,1,1]))

## init variables
init = tf.global_variables_initializer()

with tf.Session() as sess:

	## Create random input variables 
    y_pred = tf.random_uniform([20,6])
    y_true = tf.random_uniform([20,6])

	## Create deterministic input variables 
    #y_pred = tf.constant([1., 2., 3., 4., 5., 6.], shape=[20,6])
    #y_true = tf.constant([3., 4., 5., 6., 7., 8.], shape=[20,6])

    ## Create graph
    dist = loss.geodesic_loss(y_pred,y_true)

    ## Init session
    sess.run(init)

    ## Compute forward loss
    print sess.run(dist)
    #print dist.eval()

    ## Compute backward gradients
    print sess.run(tf.gradients(dist, [y_pred,y_true]))
    #print tf.gradients(dist, [y_pred,y_true])[0].eval() # grad w.r.t. y_pred
    #print tf.gradients(dist, [y_pred,y_true])[1].eval() # grad w.r.t. y_true




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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append('/vol/medic01/users/bh1511/_build/geomstats-master/')
from geomstats.special_euclidean_group import SpecialEuclideanGroup


class SE3GeodesicLoss(object):
    """docstring for SE3GeodesicLoss"""
    def __init__(self, w, op_name='GeodesicDistance'):
        super(SE3GeodesicLoss, self).__init__()

        assert w.shape != 6 , 'Weight vector must be of shape 1x6'

        self.op_name = op_name
        self.SE3_GROUP = SpecialEuclideanGroup(3)
        self.w = w
        self.SE3_GROUP.left_canonical_metric.inner_product_mat_at_identity = np.eye(6) * self.w
        self.metric = self.SE3_GROUP.left_canonical_metric

    # Python Custom Op Tensorflow Wrapper
    def py_func(self, func, inp, Tout, stateful=True, name=None, grad=None):
        """
        PyFunc defined as given by Tensorflow
    
        :param func:        Custom Function
        :param inp:         Function Inputs
        :param Tout:        Ouput Type of out Custom Function
        :param stateful:    Calculate Gradients when stateful is True
        :param name:        Name of the PyFunction
        :param grad:        Custom Gradient Function
        :return:
        """
        # Generate Random Gradient name in order to avoid conflicts with inbuilt names
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 2**32-1))
    
        # Register Tensorflow Gradient
        tf.RegisterGradient(rnd_name)(grad)
    
        # Get current graph
        g = tf.get_default_graph()
    
        # Add gradient override map
        with g.gradient_override_map({'PyFunc': rnd_name, 'PyFuncStateless': rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
    
    
    # Python Custom Op  
    def geodesic_loss(self, y_pred, y_true, name=None):
        """
        Custom Function which defines pyfunc and gradient override
        :param x:       y_pred - predicted se(3) pose
        :param y:       y_true - ground truth se(3) pose
        :param name:    Function name
        :return:        dist - geodesic distance between predicted pose and ground truth pose
        """
        with ops.name_scope(name, self.op_name, [y_pred, y_true]) as name:
            """
            Our pyfunc accepts 2 input parameters and returns 2 outputs
            Input Parameters:   y_pred, y_true
            Output Parameters:  geodesic distance, geodesic distance w.r.t. y_pred
            """
            dist , grad = self.py_func(self.riemannian_dist_grad,
                                       [y_pred, y_true],
                                       [tf.float32, tf.float32],
                                       name=name,
                                       grad=self.riemannian_grad_op)
            return dist
    
    
    # Geodesic Loss Core Function 
    def riemannian_dist_grad(self, y_pred, y_true):
        """
        Geodesic Loss Core Function 
    
        :param y_pred: y_pred
        :param y_true: y_true
        :return: dist, grad
        """
        # Geodesic Distance 
        dist = np.squeeze(self.metric.squared_dist(y_pred, y_true)).astype('float32')
    
        # d/dx (Geodesic Distance)
        tangent_vec = np.squeeze(self.metric.log(base_point=y_pred, point=y_true)).astype('float32')
        
        grad_point = - 2. * tangent_vec
    
        inner_prod_mat = np.squeeze(self.metric.inner_product_matrix(y_pred)).astype('float32')
    
        grad_point = np.repeat(np.expand_dims(grad_point,axis=1),6,axis=1)
        grad = np.sum(np.multiply(inner_prod_mat,grad_point),axis=2)
    
        sqrt_w = np.sqrt(self.w).astype('float32')
        grad = np.multiply(grad,sqrt_w)
    
        return np.sum(dist), grad
    
    
    # Geodesic Loss Gradient Function
    def riemannian_grad_op(self, op, grads, grad_glob):
        """
        Geodesic Loss Gradient Function
    
        :param op:      Operation - operation.inputs = [y_pred, y_true], operation.outputs=[dist, grad]
        :param grads:   Gradients for equation prime
        :param grad_glob: - No real use of it, but the gradient function parameter size should match op.inputs
        :return: grads * d/d_y_pred , vector of ones
        """
        # Only gradient w.r.t. y_pred is returned. 
        return grads * op.outputs[1] , tf.ones_like(op.outputs[1]) 
    







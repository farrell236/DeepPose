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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ================================================================================

import torch
from torch.autograd import Function
import numpy as np

# import sys
# sys.path.append('/vol/medic01/users/bh1511/_build/geomstats-master/')
from geomstats.special_euclidean_group import SpecialEuclideanGroup


# custom autograd function
class SE3GeodesicLoss(Function):
    def __init__(self, w):
        super(SE3GeodesicLoss, self).__init__()
        self.SE3_GROUP = SpecialEuclideanGroup(3)
        self.metric = self.SE3_GROUP.left_canonical_metric
        self.w = w

    def forward(self, inputs, targets):
        """
        PyTorch Custom Forward Function

        :param inputs:      Custom Function
        :param targets:     Function Inputs
        :return:
        """
        self.y_pred = inputs.numpy()
        self.y_true = targets.numpy()

        dist = np.squeeze(self.metric.squared_dist(self.y_pred, self.y_true))

        return torch.FloatTensor([np.mean(dist)])

    def backward(self, grad_output):
        """
        PyTorch Custom Backward Function

        :param grad_output: Gradients for equation prime
        :return:            gradient w.r.t. input
        """

        tangent_vec = self.metric.log(
            base_point=self.y_pred,
            point=self.y_true)
        tangent_vec = np.squeeze(tangent_vec).astype('float32')

        grad_point = - 2. * tangent_vec

        inner_prod_mat = self.metric.inner_product_matrix(self.y_pred)
        inner_prod_mat = np.squeeze(inner_prod_mat).astype('float32')

        grad_point = np.repeat(np.expand_dims(grad_point, axis=1), 6, axis=1)
        grad = np.sum(np.multiply(inner_prod_mat, grad_point), axis=2)

        sqrt_w = np.sqrt(self.w).astype('float32')
        grad = np.multiply(grad, sqrt_w)

        return grad_output * torch.FloatTensor(grad), None

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

from geomstats.special_euclidean_group import SpecialEuclideanGroup


# custom autograd function
class SE3GeodesicLoss(Function):
    """
    Geodesic Loss on the Special Euclidean Group SE(3), of 3D rotations
    and translations, computed as the square geodesic distance with respect
    to a left-invariant Riemannian metric.
    """
    def __init__(self, weight):

        assert weight.shape != 6, 'Weight vector must be of shape 1x6'

        self.SE3_GROUP = SpecialEuclideanGroup(3)
        self.weight = weight
        self.SE3_GROUP.left_canonical_metric.inner_product_mat_at_identity = \
            np.eye(6) * self.weight
        self.metric = self.SE3_GROUP.left_canonical_metric

    def forward(self, inputs, targets):
        """
        PyTorch Custom Forward Function

        :param inputs:      Custom Function
        :param targets:     Function Inputs
        :return:
        """
        self.y_pred = inputs.numpy()
        self.y_true = targets.numpy()

        sq_geodesic_dist = self.metric.squared_dist(self.y_pred, self.y_true)
        batch_loss = np.sum(sq_geodesic_dist)

        return torch.FloatTensor([batch_loss])

    def backward(self, grad_output):
        """
        PyTorch Custom Backward Function

        :param grad_output: Gradients for equation prime
        :return:            gradient w.r.t. input
        """

        tangent_vec = self.metric.log(
            base_point=self.y_pred,
            point=self.y_true)

        grad_point = - 2. * tangent_vec

        inner_prod_mat = self.metric.inner_product_matrix(
            base_point=self.y_pred)

        riemannian_grad = np.einsum('ijk,ik->ij', inner_prod_mat, grad_point)

        sqrt_weight = np.sqrt(self.weight)
        riemannian_grad = np.multiply(riemannian_grad, sqrt_weight)

        return grad_output * torch.FloatTensor(riemannian_grad), None

"""
Example usage of se3_geodesic_loss for PyTorch Framework

Input tensors:
    y_pred : predicted se3 pose (shape = [Nx6])
    y_true : ground truth se3 pose (shape = [Nx6])

Output tensor:
    loss   : total squared geodesic loss of all N examples

N.B. Gradient tensor (shape = [Nx6]) is only calculated w.r.t. y_pred.
Gradient w.r.t. y_true is a vector of ones.
"""

import numpy as np
import torch

from se3_geodesic_loss import SE3GeodesicLoss
from torch.autograd import Variable


N_SAMPLES = 10
SE3_DIM = 6

weight = np.ones(SE3_DIM)

# Create random input variables
y_pred = Variable(torch.rand(N_SAMPLES, SE3_DIM), requires_grad=True)
y_true = Variable(torch.rand(N_SAMPLES, SE3_DIM), requires_grad=False)

# Create deterministic input variables
'''
_y_pred = np.array([
    [-2.02058, -0.0173334, -0.556361, -0.80401, -1.64088, -0.485825],
    [-2.02058, -0.0173334, -0.556361, -0.80401, -1.64088, -0.485825]
    ])

_y_true = np.array([
    [-0.386787, 2.35808, 0.318384, -1.55862, -0.169628, 0.00206226],
    [-0.386787, 2.35808, 0.318384, -1.55862, -0.169628, 0.00206226]
    ])

y_pred = Variable(torch.FloatTensor(_y_pred), requires_grad=True)
y_true = Variable(torch.FloatTensor(_y_true), requires_grad=False)
'''

# Define loss tensor
loss = SE3GeodesicLoss(weight)(y_pred, y_true)
loss.backward()

# Print outputs
print 'y_pred:', y_pred
print 'y_true:', y_true

print 'FORWARD:', loss
print 'BACKWARD:', y_pred.grad.data

# Gradient Check
res = torch.autograd.gradcheck(
    SE3GeodesicLoss(weight),
    (y_pred, y_true),
    raise_exception=False)

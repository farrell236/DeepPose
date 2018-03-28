from se3_geodesic_loss import SE3GeodesicLoss
import numpy as np
import torch
from torch.autograd import Variable

# Init variables
y_pred = Variable(torch.rand(2, 6), requires_grad=True)
y_true = Variable(torch.rand(2, 6), requires_grad=False)

w = np.array([1, 1, 1, 1, 1, 1])

# Define loss tensor
loss = SE3GeodesicLoss(w)(y_pred, y_true)
loss.backward()

# Print outputs
print 'y_pred:', y_pred
print 'y_true:', y_true

print 'FORWARD:', loss
print 'BACKWARD:', y_pred.grad.data

# Gradient Check
# res = torch.autograd.gradcheck(
#     SE3GeodesicLoss(w),
#     (y_pred, y_true),
#     raise_exception=False)

# res should be True if the gradients are correct.
# print 'Gradient Correct:', res

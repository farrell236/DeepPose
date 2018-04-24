# DeepPose

A general Riemannian formulation of the pose estimation problem to train CNNs directly on SE(3) equipped with a left-invariant Riemannian metric.

## Getting Started

### Prerequisites

This loss function is created for TensorFlow (r1.2+) and PyTorch (v0.1+). A modified Caffe distribution of the implementation can be found [here](http://www.example.com/).

This package requires [Geomstats](https://github.com/ninamiolane/geomstats) (v1.5+) and its prerequisites.

### Usage

Git clone this repository and copy ```se3_geodesic_loss.py``` to your neural network project folder. Import it into your python project using:

```
from se3_geodesic_loss import SE3GeodesicLoss
```

Create the loss object with desired loss weights, and invoke it as part of your computational graph.

```
weight = np.array([1., 1., 1., 1., 1., 1.])

# TensorFlow
loss = SE3GeodesicLoss(weight)
geodesic_loss = loss.geodesic_loss(y_pred, y_true)

# PyTorch
loss = SE3GeodesicLoss(weight)(y_pred, y_true)
loss.backward()

```


```y_pred``` and ```y_true``` input tensors must have shape ```[Nx6]```. The gradient is calculated w.r.t. ```y_pred``` only.


## Authors & Citation

* Benjamin Hou
* Nina Miolane
* Bishesh Khanal
* Bernhard Kainz

If you like our work and found it useful for your research, please cite our paper. Thanks! :)

```
@inproceedings{hou2018computing,
  title={Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry},
  author={Hou, Benjamin and Miolane, Nina and Khanal, Bishesh and Lee, Matthew and Alansary, Amir and McDonagh, Steven and Hajnal, Jo V and Rueckert, Daniel and Glocker, Ben and Kainz, Bernhard},
  booktitle={ International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2018},
  organization={Springer}
}
```

```
@misc{miolane2018geomstats, 
  title={Geomstats: Computations and Statistics on Manifolds with Geometric Structures.}, 
  url={https://github.com/ninamiolane/geomstats}, 
  journal={GitHub}, 
  author={Miolane, Nina}, 
  year={2018}, 
  month={Feb}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

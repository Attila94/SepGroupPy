# Exploiting Learned Symmetries in Group Equivariant Convolutions

[[arXiv](https://arxiv.org/abs/2106.04914)] - ICIP 2021, by [Attila Lengyel](https://attila94.github.io) and [Jan van Gemert](http://jvgemert.github.io/index.html).

This repository contains the PyTorch implementation of separable group equivariant convolutions and the experiments described in the paper.

## Abstract
Group Equivariant Convolutions (GConvs) enable convolutional neural networks to be equivariant to various transformation groups, but at an additional parameter and compute cost. We investigate the filter parameters learned by GConvs and find certain conditions under which they become highly redundant. We show that GConvs can be efficiently decomposed into depthwise separable convolutions while preserving equivariance properties and demonstrate improved performance and data efficiency on two datasets.

## Getting started

All code and experiments have been tested with PyTorch 1.7.0.

### Separable GrouPy

`sepgroupy` implements separable group equivariant convolutions. The code is built upon [the original GrouPy repository](https://github.com/tscohen/GrouPy) by Taco. S. Cohen and [the PyTorch implementation](https://github.com/adambielski/pytorch-gconv-experiments) by Adam Bielski.

Install `sepgroupy` as follows:
```sh
git clone https://github.com/Attila94/sepgroupy
cd sepgroupy
python setup.py install
```

Then one can use separable group equivariant convolutions as regular PyTorch layers:
```python
from sepgroupy.gconv.gc_splitgconv2d import gcP4MConvP4M
layer = gcP4MConvP4M(in_channels, out_channels, kernel_size, stride, padding, bias)
```

See also the models used in the [experiments](experiments).

### Experiments

#### Rotated MNIST
Download and unpack the Rotated MNIST dataset:
```sh
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
unzip mnist_rotation_new.zip 
rm mnist_rotation_new.zip
```

Train a gc-P4 CNN:
```sh
python main.py --planes 158 --equivariance 'P4' --separable 'gc'
```


#### CIFAR10
Download and unpack prewhitened CIFAR10:
```sh
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
```

Train a gc-P4M-ResNet44:
```sh
python main.py --dataset_path 'path/to/cifar' --model 'resnet44_gcp4m'
```

## Computational complexity

The tables below provide an overview of the models used for the experiments in the paper in terms of their test error, parameter count, Multiply-Accumulate operations (MACs) and GPU memory footprint. Memory footprint is calculated for a forward+backward pass using a batch size of 2. *w* denotes the network width in number of channels.

| Model            | *w*  | Test error           | Param.     | MACs       | GPU memory |
| ---------------- | ---- | -------------------- | ---------- | ---------- | ---------- |
| Z2CNN            | 20   | 5.20 +- 0.110     | 25.21 k    | 2.98 M     | 0.84 MB    |
| *c*-Z2CNN        | 57   | 4.64 +- 0.126     | 25.60 k    | 4.14 M     | 2.60 MB    |
| P4CNN            | 10   | 2.23 +- 0.061     | 24.81 k    | 11.67 M    | 2.03 MB    |
| *g*-P4CNN-small  | 10   | 2.60 +- 0.098     | 8.91 k     | 4.37 M     | 5.23 MB    |
| *g*-P4CNN        | 17   | 1.97 +- 0.044     | 25.26 k    | 12.34 M    | 13.84 MB   |
| *gc*-P4CNN-small | 10   | 2.88 +- 0.169     | **3.42 k** | **1.80 M** | 1.88 MB    |
| *gc*-P4CNN       | 30   | **1.74 +- 0.070** | 24.64 k    | 13.01 M    | 5.83 MB    |

| Model               | CIFAR10 test err. | CIFAR10+ test err. | Param.     | MACs       | GPU memory   |
| ------------------- | ----------------- | ------------------ | ---------- | ---------- | ------------ |
| ResNet44            | 13.10             | 7.66               | 2.64 M     | **0.39 G** | **23.19 MB** |
| *p4m*-ResNet44      | 8.06              | 5.78               | 2.62 M     | 3.08 G     | 141.77 MB    |
| *g*-*p4m*-ResNet44  | 7.60              | 6.09               | **1.78 M** | 2.07 G     | 1141.31 MB   |
| *gc*-*p4m*-ResNet44 | **6.72**          | **5.43**           | 1.88 M     | 2.17 G     | 197.84 MB    |

> **Note on *g*-separable GConvs**: due to the PyTorch implementation of depthwise convolutions, intermediate feature maps and filter weights for *g*-GConvs need to be reshaped using  `.reshape` instead of  `.view`, and as such extra tensor copies are generated. This results in additional computational overhead in terms of GPU memory usage and runtime. All experiments have therefore been run using a "naive" implementation, where the filter weights are first precomputed and convolutions are performed in a non-separable way, resulting in the exact same output. This "naive" mode can be enabled using `separable=False` and has the same computational cost as regular GConvs. *gc*-GConvs are note affected by this and are more efficient, both theoretically and in practice.

## Citation

If you find this repository useful for your work, please cite as follows:

```
@article{lengyel2021exploiting,
  title={Exploiting Learned Symmetries in Group Equivariant Convolutions},
  author={Attila Lengyel and Jan C. van Gemert},
  year={2021},
  eprint={2106.04914},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```


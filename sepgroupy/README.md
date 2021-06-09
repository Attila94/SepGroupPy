# SepGrouPy

SepGrouPy is the depthwise separable PyTorch implementation of GrouPy, as introduced by[\[Cohen & Welling, 2016\]](#gcnn). For the full original readme see [GrouPy](https://github.com/tscohen/GrouPy) and extended by [Adam Bielsky](https://github.com/adambielski/pytorch-gconv-experiments).

## Setup

SepGrouPy has been tested and used with PyTorch 1.7.0. To use, install the latest version of [PyTorch](https://pytorch.org), clone the latest version of this github repository and run setup.py:

```
$ python setup.py install
```

## References

1. <a name="gcnn"></a> T.S. Cohen, M. Welling, [Group Equivariant Convolutional Networks](http://www.jmlr.org/proceedings/papers/v48/cohenc16.pdf). Proceedings of the International Conference on Machine Learning (ICML), 2016.

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
from sepgroupy.gconv.make_gconv_indices import *
from sepgroupy.gconv.tf import *

# Share same blueprint over input group dimension
# 1. Pointwise convolution over group dimension to get weighted sum of input feature maps.
# 2. Regular 2D convolution with filter size [n_o*g_o, n_i, k, k] where g_o contains rotated copies.

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}


def trans_filter(w, inds):
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                               inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()

def pre_trans_filter(dw,pw,inds):

    # construct full filter bank from blueprints and group weights
    w = torch.mul(pw,dw)
    w_transformed = trans_filter(w, inds)
    return w_transformed

class SplitGConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 input_stabilizer_size=1, output_stabilizer_size=4, separable=True):
        super(SplitGConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.separable = separable

        if self.input_stabilizer_size == 1:
            # input layer
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, 1, *kernel_size))
        else:
            # hidden layer
            self.depthwise_weight = Parameter(torch.Tensor(
                out_channels, 1, 1, *kernel_size))
            self.pointwise_weight = Parameter(torch.Tensor(
                out_channels, in_channels, self.input_stabilizer_size, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.input_stabilizer_size == 1:
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.depthwise_weight.data.uniform_(-stdv, stdv)
            self.pointwise_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        # Reshape input
        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels * self.input_stabilizer_size,
                           input_shape[-2], input_shape[-1])

        if self.input_stabilizer_size == 1:
            # Nothing to decompose for input layer
            tw = trans_filter(self.weight, self.inds)
            tw_shape = (self.out_channels * self.output_stabilizer_size,
                        self.in_channels * self.input_stabilizer_size,
                        self.ksize, self.ksize)
            tw = tw.view(tw_shape)

            y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                         padding=self.padding)
        elif self.separable:
            # Not input layer, separable convolution

            tw_depthwise = trans_filter_d(self.depthwise_weight, self.input_stabilizer_size)
            tw_pointwise = trans_filter_p(self.pointwise_weight)

            # Pointwise convolution
            # Reshape weights
            tw_shape = (self.out_channels * self.output_stabilizer_size,
                        self.in_channels * self.input_stabilizer_size, 1, 1)
            tw = tw_pointwise.view(tw_shape)
            # Convolve
            y = F.conv2d(input, weight=tw, bias=None)

            # Spatial convolution
            # Reshape weights
            tw_shape = (self.out_channels * self.output_stabilizer_size,
                        1, self.ksize, self.ksize)
            tw = tw_depthwise.view(tw_shape)
            # Convolve
            y = F.conv2d(y, weight=tw, bias=None, stride=self.stride, padding=self.padding,
                         groups=self.out_channels * self.output_stabilizer_size)

        else:
            # Not input layer, not separable convolution
            tw = pre_trans_filter(self.depthwise_weight, self.pointwise_weight, self.inds)
            tw_shape = (self.out_channels * self.output_stabilizer_size,
                        self.in_channels * self.input_stabilizer_size,
                        self.ksize, self.ksize)
            tw = tw.view(tw_shape)

            y = F.conv2d(input, weight=tw, bias=None, stride=self.stride, padding=self.padding)

        # Reshape output
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # Apply bias
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class gcP4ConvZ2(SplitGConv2D):
    # Same as regular P4ConvZ2
    def __init__(self, *args, **kwargs):
        super(gcP4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class gcP4ConvP4(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(gcP4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)


class gcP4MConvZ2(SplitGConv2D):
    # Same as regular P4MConvZ2
    def __init__(self, *args, **kwargs):
        super(gcP4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class gcP4MConvP4M(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(gcP4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)

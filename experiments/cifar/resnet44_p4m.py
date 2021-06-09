#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:51:47 2020 by Attila Lengyel - attila@lengyel.nl
"""

from torchvision.models.resnet import ResNet

import torch
import torch.nn as nn

from sepgroupy.gconv.splitgconv2d import P4MConvZ2, P4MConvP4M
from sepgroupy.gconv.g_splitgconv2d import gP4MConvP4M
from sepgroupy.gconv.gc_splitgconv2d import gcP4MConvP4M


def conv3x3(layer, in_planes, out_planes, stride=1):

    """3x3 convolution with padding"""
    return layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""
    return P4MConvP4M(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64,
                 conv_layer=None, norm_layer=None):
        super(BasicBlock, self).__init__()

        if conv_layer is None: conv_layer = P4MConvP4M
        if norm_layer is None: norm_layer = nn.BatchNorm3d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(conv_layer, inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(conv_layer, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        return x

class CustomResNet(ResNet):

    def __init__(self, block, layers, channels, num_classes=1000, sep=None,
                 zero_init_residual=False, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None: norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        conv_layer = {None: P4MConvP4M, 'g': gP4MConvP4M, 'gc': gcP4MConvP4M}

        self.inplanes = channels[0]
        self.base_width = width_per_group
        self.conv1 = P4MConvZ2(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(channels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channels[0], layers[0], conv_layer[sep])
        self.layer2 = self._make_layer(block, channels[1], layers[1], conv_layer[sep], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], conv_layer[sep], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2] * block.expansion * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual branch starts
        # with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, conv_layer, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, conv_layer, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                                conv_layer=conv_layer, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        n, nc, ns, nx, ny = x.shape
        x = x.view(n,nc*ns,nx,ny)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def p4mresnet44(**kwargs):
    return CustomResNet(BasicBlock, [7,7,7], [11,23,45], **kwargs)

def gp4mresnet44(**kwargs):
    return CustomResNet(BasicBlock, [7,7,7], [19,38,76], sep='g', **kwargs)

def gcp4mresnet44(**kwargs):
    return CustomResNet(BasicBlock, [7,7,7], [28,56,112], sep='gc', **kwargs)

def getMaxCudaMem(model, in_shape):
    # Measure memory use
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    x = torch.rand(in_shape).cuda()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.max_memory_allocated()

    # Perform forward + backward pass
    y = m(x)
    label = torch.tensor([0, 0]).cuda()
    loss = criterion(y, label)
    loss.backward()

    mem = (torch.cuda.max_memory_allocated() - s) / 1024**2
    print('Max memory used: {:.2f} MB'.format(mem))


if __name__ == '__main__':
    from torchinfo import summary

    m = p4mresnet44(num_classes=10)
    print(m)
    summary(m, (1,3,32,32), device='cpu', col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
    getMaxCudaMem(m, (2,3,32,32))

    print('---')

    m = gp4mresnet44(num_classes=10)
    print(m)
    summary(m, (1,3,32,32), device='cpu', col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
    getMaxCudaMem(m, (2,3,32,32))

    print('---')

    m = gcp4mresnet44(num_classes=10)
    print(m)
    summary(m, (1,3,32,32), device='cpu', col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
    getMaxCudaMem(m, (2,3,32,32))

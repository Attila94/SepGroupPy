#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:51:47 2020 by Attila Lengyel - attila@lengyel.nl
"""

from torchvision.models.resnet import ResNet, conv1x1

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, separable=False):

    """3x3 convolution with padding"""
    if separable:  # PW+DW
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                             nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False, groups=out_planes))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64, norm_layer=None, separable=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, separable=separable)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, separable=separable)
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

    def __init__(self, block, layers, channels, num_classes=1000,
                 zero_init_residual=False, width_per_group=64, norm_layer=None, separable=False):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = channels[0]
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(channels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channels[0], layers[0], separable=separable)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, separable=separable)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, separable=separable)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2] * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, separable=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, norm_layer, separable=separable))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                                norm_layer=norm_layer, separable=separable))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def resnet44(channels=[32,64,128], **kwargs):
    return CustomResNet(BasicBlock, [7,7,7], channels, **kwargs)


if __name__ == '__main__':
    from torchinfo import summary

    m = resnet44(num_classes=10)
    print(m)
    summary(m, (1,3,32,32), device='cpu', col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))

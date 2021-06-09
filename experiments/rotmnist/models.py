import torch.nn as nn
import torch.nn.functional as F

from sepgroupy.gconv.pooling import GroupCosetMaxPool, GroupMaxPool2d
from sepgroupy.gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from sepgroupy.gconv.g_splitgconv2d import gP4ConvP4
from sepgroupy.gconv.gc_splitgconv2d import gcP4ConvP4

class Net(nn.Module):

    def __init__(self, planes, in_conv, hidden_conv, bn, mp, gmp,
                 num_classes=10):
        super(Net, self).__init__()
        self.conv1 = in_conv(1, planes, kernel_size=3)
        self.conv2 = hidden_conv(planes, planes, kernel_size=3, separable=False)
        self.conv3 = hidden_conv(planes, planes, kernel_size=3, separable=False)
        self.conv4 = hidden_conv(planes, planes, kernel_size=3, separable=False)
        self.conv5 = hidden_conv(planes, planes, kernel_size=3, separable=False)
        self.conv6 = hidden_conv(planes, planes, kernel_size=3, separable=False)
        self.conv7 = hidden_conv(planes, planes, kernel_size=4, separable=False)

        self.fc = nn.Linear(planes, num_classes)

        self.bn1 = bn(planes)
        self.bn2 = bn(planes)
        self.bn3 = bn(planes)
        self.bn4 = bn(planes)
        self.bn5 = bn(planes)
        self.bn6 = bn(planes)
        self.bn7 = bn(planes)

        self.mp = mp(2)
        self.gmp = gmp() if gmp is not None else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        if self.gmp:
            x = self.gmp(x)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def CNN(planes, equivariance='Z2', separable=None, num_classes=10):

    assert equivariance in ['Z2', 'P4'], 'Must be one of Z2, P4'
    assert separable in [None, 'g', 'gc'], 'Must be one of None, g, gc. g or gc can be used interchangeably for Z2.'

    conv_layer = {None: P4ConvP4,
                  'g': gP4ConvP4,
                  'gc': gcP4ConvP4}

    if equivariance == 'Z2':
        in_conv = nn.Conv2d
        if separable is None:
            hidden_conv = nn.Conv2d
        else:
            # Define separable convolution (PW+DW)
            def sep_conv(in_channels, out_channels, kernel_size):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels))
            hidden_conv = sep_conv
        bn = nn.BatchNorm2d
        mp = nn.MaxPool2d
        gmp = None
    else:
        in_conv = P4ConvZ2
        hidden_conv = conv_layer[separable]
        bn = nn.BatchNorm3d
        mp = GroupMaxPool2d
        gmp = GroupCosetMaxPool

    return Net(planes, in_conv, hidden_conv, bn, mp, gmp, num_classes)


if __name__ == '__main__':
    from torchinfo import summary
    import torch

    m = CNN(17, equivariance='P4', separable='g').cuda()
    print(m)
    summary(m, (2,1,28,28), col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))

    # Measure memory use
    criterion = nn.CrossEntropyLoss()
    x = torch.rand(2,1,28,28).cuda()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.max_memory_allocated()

    # Perform forward + backward pass
    y = m(x)
    label = torch.tensor([0, 0]).cuda()
    loss = criterion(y, label)
    loss.backward()

    mem = (torch.cuda.max_memory_allocated() - s) / 1024**2
    print('Max memory used: {:.2f} MB'.format(mem))

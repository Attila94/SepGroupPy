import numpy as np
import torch
from torch.autograd import Variable
from sepgroupy.gconv.splitgconv2d import P4ConvZ2, P4MConvZ2, P4ConvP4, P4MConvP4M

def test_p4_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4FuncArray
    import sepgroupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            P4ConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            P4ConvP4(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p4m_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4MFuncArray
    import sepgroupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            P4MConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            P4MConvP4M(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )

def test_gp4_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4FuncArray
    from sepgroupy.gconv.g_splitgconv2d import gP4ConvZ2, gP4ConvP4
    import sepgroupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            gP4ConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            gP4ConvP4(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_gp4m_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4MFuncArray
    from sepgroupy.gconv.g_splitgconv2d import gP4MConvZ2, gP4MConvP4M
    import sepgroupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            gP4MConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            gP4MConvP4M(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )

def test_gcp4_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4FuncArray
    from sepgroupy.gconv.gc_splitgconv2d import gcP4ConvZ2, gcP4ConvP4
    import sepgroupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            gcP4ConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            gcP4ConvP4(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_gcp4m_net_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4MFuncArray
    from sepgroupy.gconv.gc_splitgconv2d import gcP4MConvZ2, gcP4MConvP4M
    import sepgroupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[
            gcP4MConvZ2(in_channels=1, out_channels=2, kernel_size=3),
            gcP4MConvP4M(in_channels=2, out_channels=3, kernel_size=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_g_z2_conv_equivariance():
    from sepgroupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
    import sepgroupy.garray.C4_array as c4a
    import sepgroupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[P4ConvZ2(1, 2, 3)],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )

    return check_equivariance(
        im=im,
        layers=[P4MConvZ2(1, 2, 3)],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_p4_p4_conv_equivariance():
    from sepgroupy.gfunc import P4FuncArray
    import sepgroupy.garray.C4_array as c4a

    im = np.random.randn(1, 1, 4, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[P4ConvP4(1, 2, 3)],
        input_array=P4FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p4m_p4m_conv_equivariance():
    from sepgroupy.gfunc import P4MFuncArray
    import sepgroupy.garray.D4_array as d4a

    im = np.random.randn(1, 1, 8, 11, 11).astype('float32')
    return check_equivariance(
        im=im,
        layers=[P4MConvP4M(1, 2, 3)],
        input_array=P4MFuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def check_equivariance(im, layers, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im)
    g = point_group.rand()
    gf = g * f
    im1 = gf.v
    # Apply layers to both images
    im = Variable(torch.Tensor(im))
    im1 = Variable(torch.Tensor(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(fmap1.data.numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = fmap.data.numpy()
    return np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)


if __name__ == '__main__':
    print('P4:', test_p4_net_equivariance())
    print('P4M:', test_p4m_net_equivariance())
    print('gP4:', test_gp4_net_equivariance())
    print('gP4M:', test_gp4m_net_equivariance())
    print('gcP4:', test_gcp4_net_equivariance())
    print('gcP4M:', test_gcp4m_net_equivariance())

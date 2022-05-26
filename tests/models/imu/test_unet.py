import numpy as np
import torch
from openpack_torch.models.imu.unet import (DownBlock, UNet, UNetDecoder,
                                            UNetEncoder, UpBlock)


def test_DownBlock__01():
    in_ch, out_ch = 32, 64

    net = DownBlock(in_ch, out_ch)
    print(net)


def test_UpBlock__01():
    in_ch, out_ch = 64, 32
    batch_size = 5
    seq_len = 100

    # -- Model --
    net = UpBlock(in_ch, out_ch)
    print(net)

    # -- Forward --
    x = torch.empty((batch_size, in_ch, seq_len, 1)).normal_()
    x_skip = torch.empty((batch_size, in_ch // 2, seq_len * 2, 1)).normal_()

    x = x.to(torch.device("cpu"), torch.float)
    print(f"x={x.size()}")

    y = net(
        x.to(torch.device("cpu"), torch.float),
        x_skip.to(torch.device("cpu"), torch.float),
    )
    print(f"y={y.size()}")
    np.testing.assert_array_equal(
        y.size(), (batch_size, in_ch // 2, seq_len * 2, 1))


def test_UNetEncoder__01():
    batch_size = 5
    seq_len = 1800
    in_ch = 32
    depth = 2

    # -- Model --
    net = UNetEncoder(in_ch, depth=depth)
    net.to(torch.device("cpu"), torch.float)
    print(net)

    # -- Forward --
    x = torch.empty((batch_size, in_ch, seq_len, 1)).normal_()
    x = x.to(torch.device("cpu"), torch.float)
    print(f"x={x.size()}")

    y, skip_connections = net(x)
    print(f"y={y.size()}")
    for i in range(len(skip_connections)):
        print(f"skip_connections[{i}]={skip_connections[i].size()}")

    assert y.size()[0] == batch_size
    assert y.size()[1] == in_ch * (2**depth)
    assert y.size()[2] - (seq_len // (2**depth)) <= 1
    assert y.size()[3] == 1


def test_UNetDecoder__01():
    batch_size = 5
    seq_len_init = 1800
    in_ch_init = 32
    depth = 2

    # -- Model --
    net = UNetDecoder(in_ch_init, depth=depth)
    net.to(torch.device("cpu"), torch.float)
    print(net)

    # -- Forward --
    x_skips = []
    in_ch, seq_len = in_ch_init, seq_len_init
    for i in range(depth):
        tmp = torch.empty((batch_size, in_ch, seq_len, 1)).normal_()
        x_skips.append(tmp.to(torch.device("cpu"), torch.float))
        in_ch *= 2
        seq_len //= 2
        print(f"x_skips[{i}]={x_skips[i].size()}")
    x = torch.empty((batch_size, in_ch, seq_len, 1)).normal_()
    x = x.to(torch.device("cpu"), torch.float)
    print(f"x={x.size()}")

    y = net(x, x_skips)
    print(f"y={y.size()}")

    np.testing.assert_array_equal(
        y.size(), (batch_size, in_ch_init, seq_len_init, 1))


def test_UNet__01():
    batch_size = 5
    seq_len = 1800
    in_ch = 6
    depth = 5

    num_classes = 10

    # -- Model --
    net = UNet(in_ch, num_classes=num_classes, depth=depth)
    net.to(torch.device("cpu"), torch.float)
    print(net)

    # -- Forward --
    x = torch.empty((batch_size, in_ch, seq_len, 1)).normal_()
    x = x.to(torch.device("cpu"), torch.float)
    print(f"x={x.size()}")

    y = net(x)
    print(f"y={y.size()}")

    np.testing.assert_array_equal(
        y.size(), (batch_size, num_classes, seq_len, 1))

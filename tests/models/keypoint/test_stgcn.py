import numpy as np
import torch
from openpack_torch.models.keypoint.graph import get_adjacency_matrix
from openpack_torch.models.keypoint.stgcn import (SpatialGraphConvLayer,
                                                  STConvBlock, STGCN4Seg,
                                                  TemporalConvLayer)


def test_SpatialGraphConvLayer__01():
    in_channels = 32
    out_channels = 64
    Ks = 3
    batch_size = 5
    num_frames = 80
    num_vertex = 25

    net = SpatialGraphConvLayer(in_channels, out_channels, Ks)
    net = net.to(dtype=torch.float)
    print("")
    print(net)

    A = get_adjacency_matrix(layout="NTU-RGBD", hop_size=Ks - 1)
    A = torch.from_numpy(A).to(dtype=torch.float)
    print("A:", A.size())

    # -- forward --
    x = torch.empty(
        (batch_size,
         in_channels,
         num_frames,
         num_vertex)).normal_()
    x = x.to(dtype=torch.float)
    print("x:", x.size())

    y = net(x, A)
    print("y:", y.size())
    np.testing.assert_array_equal(
        y.shape, (batch_size, out_channels, num_frames, num_vertex))


def test_TmporalGraphConvLayer__01():
    in_channels = 32
    Kt = 9
    batch_size = 5
    seq_len = 80
    num_vertex = 25

    net = TemporalConvLayer(in_channels, Kt)
    net = net.to(dtype=torch.float)
    print("")
    print(net)

    # -- forward --
    x = torch.empty(
        (batch_size,
         in_channels,
         seq_len,
         num_vertex)).normal_()
    x = x.to(dtype=torch.float)
    print("x:", x.size())

    y = net(x)
    print("y:", y.size())
    np.testing.assert_array_equal(y.shape, x.shape)


def test_STConvBlock__01():
    batch_size = 5
    # num_ch = 3  # 3D Data (x, y, z)
    in_channels = 32
    out_channels = 64
    num_frames = 80
    num_vertex = 25

    # -- Build Model --
    Kt, Ks = 3, 3
    A = get_adjacency_matrix(layout="NTU-RGBD", hop_size=Ks - 1)
    A = torch.from_numpy(A).to(dtype=torch.float)
    print("A:", A.size())

    net = STConvBlock(
        in_channels,
        out_channels,
        Kt,
        Ks,
        num_vertex=num_vertex,
    ).to(dtype=torch.float)
    print()
    print(net)

    # -- forward --
    x = torch.empty(
        (batch_size,
         in_channels,
         num_frames,
         num_vertex)).normal_()
    x = x.to(dtype=torch.float)
    print("x:", x.size())

    y = net(x, A)
    print("y:", y.size())
    np.testing.assert_array_equal(
        y.shape, (batch_size, out_channels, num_frames, num_vertex))


# -----------------------------------------------------------------------------
def test_STGCN4Seg__01():
    batch_size = 5
    num_frames = 15 * 20
    num_vertex = 17

    in_channels = 2  # 2D Data (x, y, z)
    num_classes = 10

    Ks = 3
    Kt = 9

    A = get_adjacency_matrix(layout="MSCOCO", hop_size=Ks - 1)
    print("A:", A.shape)

    model = STGCN4Seg(in_channels, num_classes, Ks=Ks, Kt=Kt, A=A)
    print(model)

    # -- Forward --

    x = torch.empty(
        (batch_size,
         in_channels,
         num_frames,
         num_vertex)).normal_()
    print("x:", x.size())

    y = model(x)
    print("y:", y.size())

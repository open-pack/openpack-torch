import numpy as np
import torch
from openpack_torch.models.imu.deep_conv_lstm import (DeepConvLSTM,
                                                      DeepConvLSTMSelfAttn)


def test_DeepConvLSTM__01():
    batch_size = 5
    seq_len = 1800
    in_ch = 6

    num_classes = 10

    # -- Model --
    net = DeepConvLSTM(in_ch, num_classes=num_classes)
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


def test_DeepConvLSTMSelfAttn__01():
    batch_size = 5
    seq_len = 1800
    in_ch = 6

    num_classes = 10

    # -- Model --
    net = DeepConvLSTMSelfAttn(in_ch, num_classes=num_classes)
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

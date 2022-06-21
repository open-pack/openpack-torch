from pathlib import Path

import numpy as np
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_torch.data.datasets import OpenPackImu, OpenPackKeypoint


# -----------------------------------------------------------------------------
def test_OpenPackImu_01():
    rootdir = Path(__file__).parents[2] / "samples/openpack/vX.X.X"
    user_session = (("U0102", "S0500"), )
    imu_nodes = ("atr01",)

    dataset = OpenPackImu(rootdir, user_session, imu_nodes=imu_nodes)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}]")
        np.testing.assert_array_equal(x.size(), (3, 1800, 1))


def test_OpenPackImu_02():
    """ submission = True """
    rootdir = Path(__file__).parents[2] / "samples/openpack/vX.X.X"
    user_session = (("U0102", "S0500"), )
    imu_nodes = ("atr01", "atr01")

    dataset = OpenPackImu(
        rootdir,
        user_session,
        imu_nodes=imu_nodes,
        submission=True)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        t_set = set(t.numpy().ravel())
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}][set={t_set}]")

        np.testing.assert_array_equal(x.size(), (6, 1800, 1))
        assert t_set <= set(
            [0, OPENPACK_OPERATIONS.get_ignore_class_index()])

# -----------------------------------------------------------------------------


def test_OpenPackKeypoint_01():
    rootdir = Path(__file__).parents[2] / "samples/openpack/vX.X.X"
    user_session = (("U0102", "S0500"), )
    # rootdir = Path(__file__).parents[1] / "sample-data/openpack/"
    # user_session = ((0, 100), )
    keypoint_type = "mmpose-hrnet-w48-posetrack18-384x288-posewarper-stage2/single"

    dataset = OpenPackKeypoint(
        rootdir,
        user_session,
        keypoint_type=keypoint_type,
        window=30)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}]")
        np.testing.assert_array_equal(x.size(), (2, 30, 17))

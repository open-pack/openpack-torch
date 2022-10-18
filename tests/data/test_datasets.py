import copy
from pathlib import Path

import numpy as np
import openpack_toolkit as optk
import pytest
from omegaconf import OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_torch.data.datasets import OpenPackImu, OpenPackKeypoint


@pytest.fixture
def opcfg():
    rootdir = Path(__file__).parents[2] / "samples/openpack/${.version}"

    cfg = OmegaConf.create({
        "path": {
            "openpack": {
                "version": optk.SAMPLE_DATASET_VERSION,
                "rootdir": str(rootdir),
            }
        },
        "dataset": {
            "stream": None,
            "annotation": optk.configs.datasets.annotations.ACTIVITY_1S_ANNOTATION,
        }
    })
    return cfg

# -----------------------------------------------------------------------------


def test_OpenPackImu_01(opcfg):
    user_session = (("U0102", "S0500"), )
    opcfg.dataset.stream = optk.configs.datasets.streams.ATR_ACC_STREAM

    dataset = OpenPackImu(opcfg, user_session)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}]")
        np.testing.assert_array_equal(x.size(), (3 * 4, 1800, 1))


def test_OpenPackImu_02(opcfg):
    """ submission = True """
    user_session = (("U0102", "S0500"), )
    opcfg.dataset.stream = optk.configs.datasets.streams.ATR_ACC_STREAM

    dataset = OpenPackImu(
        opcfg,
        user_session,
        submission=True)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        t_set = set(t.numpy().ravel())
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}][set={t_set}]")

        np.testing.assert_array_equal(x.size(), (3 * 4, 1800, 1))
        assert t_set <= set(
            [0, OPENPACK_OPERATIONS.get_ignore_class_index()])

# -----------------------------------------------------------------------------


def test_OpenPackKeypoint_01(opcfg):
    user_session = (("U0102", "S0500"), )
    opcfg.dataset.stream = optk.configs.datasets.streams.KINECT_2D_KPT_STREAM

    dataset = OpenPackKeypoint(
        copy.deepcopy(opcfg),
        user_session,
        window=30)
    print("Dataset:", dataset)

    for index in range(0, min(10, len(dataset))):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}]")
        np.testing.assert_array_equal(x.size(), (2, 30, 17))

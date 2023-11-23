import copy
from pathlib import Path

import numpy as np
import openpack_toolkit as optk
import pytest
import yaml
from omegaconf import OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS, ActSet

from openpack_torch.data.datasets import Kinect2dKptDataset, OpenPackImu

ACTSET_OPENPACK_OPERATIONS = ActSet(OPENPACK_OPERATIONS)


@pytest.fixture
def opcfg():
    rootdir = Path(__file__).parents[2] / "samples/openpack/${.version}"

    cfg = OmegaConf.create(
        {
            "path": {
                "openpack": {
                    "version": optk.SAMPLE_DATASET_VERSION,
                    "rootdir": str(rootdir),
                }
            },
            "dataset": {
                "stream": None,
                "annotation": {
                    "name": "openpack-operations-1hz-annotation",
                    "metadata": {
                        "labels": {
                            "type": "annotation/operation",
                            "version": "v3.5.0",
                            "dependency": "openpack-operations",
                            "resolution": "1Hz",
                        }
                    },
                    "spec": {
                        "path": {
                            "dir": "${path.openpack.rootdir}/${user.name}/annotation/openpack-operations-1hz",
                            "fname": "${session}.csv",
                        }
                    },
                },
            },
        }
    )
    return cfg


_ATR02_IMU_STREAM_YAML = """
kind: dataset/stream/multimodal
name: atr02-iot
mdetadata:
  labels:
    app: openpack-benchmarks
    version: 1.0.0
    multimodal: true
spec:
  imu:
    kind: dataset/stream/imu
    path:
      dir: ${path.openpack.rootdir}/${user.name}/atr/${device}
      fname: ${session}.csv
      full_path: ${.dir}/${.fname}
    devices:
      - "atr01"
      - "atr02"
      - "atr03"
      - "atr04"
    acc: true
    gyro: false
    quat: false
  iot:
    kind: dataset/stream/iot
    spec:
      anchor:
        dim: 2
        linked_class:
          0: 6
          1: 6
      devices:
        kind: dataset/stream/multimodal
        spec:
          ht:
            kind: dataset/stream
            spec:
              path:
                dir: ${path.openpack.rootdir}/${user.name}/system/ht
                fname: ${session}.csv
          printer:
            kind: dataset/stream
            spec:
              path:
                dir: ${path.openpack.rootdir}/${user.name}/system/printer
                fname: ${session}.csv
"""

_KINECT_2D_KPT_STREAM_YAML = """
kind: dataset/stream/multimodal
name: k4a-2d-kpt
mdetadata:
  labels:
    app: openpack-benchmarks
    version: 1.0.0
    multimodal: true
spec:
  kinect2dKpt:
    kind: dataset/stream/keypoint
    spec:
      poseEstimationModel: mmpose-hrnet-w48-posetrack18-384x288-posewarper-stage2
      path:
        dir: ${path.openpack.rootdir}/${user.name}/kinect/2d-kpt/${..poseEstimationModel}/single
        fname: ${session}.json
      stats:
        mean:
          x: 631.23
          y: 523.68
        std:
          x: 100.98
          y: 111.06
      frame_rate: 15
      nodes:
        0: nose
        1: left_eye
        2: right_eye
        3: left_ear
        4: right_ear
        5: left_shoulder
        6: right_shoulder
        7: left_elbow
        8: right_elbow
        9: left_wrist
        10: right_wrist
        11: left_hip
        12: right_hip
        13: left_knee
        14: right_knee
        15: left_ankle
        16: right_ankle
"""


@pytest.fixture
def atr02_imu_stream_cfg():
    cfg = yaml.safe_load(_ATR02_IMU_STREAM_YAML)
    return OmegaConf.create(cfg)


@pytest.fixture
def kinect_2d_kpt_stream_cfg():
    cfg = yaml.safe_load(_KINECT_2D_KPT_STREAM_YAML)
    return OmegaConf.create(cfg)


# -----------------------------------------------------------------------------


def test_OpenPackImu_01(opcfg, atr02_imu_stream_cfg):
    user_session = (("U0209", "S0500"),)
    opcfg.dataset.stream = atr02_imu_stream_cfg

    dataset = OpenPackImu(opcfg, user_session)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        t_set = set(t.numpy().ravel())
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}][set={t_set}],"
            f"ts={ts.size()}[{ts.dtype}]"
        )
        np.testing.assert_array_equal(x.size(), (3 * 4, 1800, 1))


def test_OpenPackImu_02(opcfg, atr02_imu_stream_cfg):
    """submission = True"""
    user_session = (("U0209", "S0500"),)
    opcfg.dataset.stream = atr02_imu_stream_cfg

    dataset = OpenPackImu(opcfg, user_session, submission=True)
    print("Dataset:", dataset)

    for index in range(len(dataset)):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        t_set = set(t.numpy().ravel())
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}][set={t_set}],"
            f"ts={ts.size()}[{ts.dtype}]"
        )

        np.testing.assert_array_equal(x.size(), (3 * 4, 1800, 1))
        assert t_set <= set([0, ACTSET_OPENPACK_OPERATIONS.get_ignore_class_index()])


# -----------------------------------------------------------------------------


def test_Kinect2dKptDataset_01(opcfg, kinect_2d_kpt_stream_cfg):
    user_session = (("U0209", "S0500"),)
    opcfg.dataset.stream = kinect_2d_kpt_stream_cfg

    dataset = Kinect2dKptDataset(copy.deepcopy(opcfg), user_session, window=30)
    print("Dataset:", dataset)

    for index in range(0, min(10, len(dataset))):
        batch = dataset.__getitem__(index)
        x, t, ts = batch["x"], batch["t"], batch["ts"]
        print(
            f"index={index}: x={x.size()}[{x.dtype}], t={t.size()}[{t.dtype}],"
            f"t={ts.size()}[{ts.dtype}]"
        )
        np.testing.assert_array_equal(x.size(), (3, 30, 17))

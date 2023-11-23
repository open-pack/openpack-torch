from pathlib import Path
from typing import Dict, Optional

import openpack_toolkit as optk
import pytest
import yaml
from omegaconf import OmegaConf

from openpack_torch.data.datamodule import OpenPackBaseDataModule
from openpack_torch.data.datasets import OpenPackImu as OpenPackImuDataset

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
"""


@pytest.fixture
def atr02_imu_stream_cfg():
    cfg = yaml.safe_load(_ATR02_IMU_STREAM_YAML)
    return OmegaConf.create(cfg)


@pytest.fixture
def opcfg(atr02_imu_stream_cfg):
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
                "stream": atr02_imu_stream_cfg,
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
                "split": {
                    "name": "unit-test",
                    "train": [("U0209", "S0500")],
                    "val": [("U0209", "S0500")],
                    "test": [("U0209", "S0500")],
                    "submission": [("U0209", "S0500")],
                },
            },
            "train": {
                "window": 30 * 60,
                "batch_size": 32,
                "num_workers": 3,
                "random_crop": True,
                "debug": {
                    "batch_size": 2,
                },
            },
            "debug": False,
        }
    )
    return cfg


# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage, debug",
    (
        (None, True),
        (None, False),
        ("fit", False),
        ("validate", False),
        ("test", False),
        ("submission", False),
        ("test-on-submission", False),
    ),
)
def test_OpenPackBaseDataModule__01(opcfg, stage, debug):
    class TestDataModule(OpenPackBaseDataModule):
        dataset_class = OpenPackImuDataset

        def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
            kwargs = {
                "window": self.cfg.train.window,
                "debug": self.cfg.debug,
            }
            return kwargs

    # -- Init datamodule --
    opcfg.debug = debug
    datamodule = TestDataModule(opcfg)
    print("datamodule:", datamodule)

    # -- Load Datasets --
    print("setup train dataset")
    datamodule.setup()
    datamodule.setup(stage=stage)

    # -- Get Dataloaders --
    if stage == "fit":
        dloader = datamodule.train_dataloader()
        print("train dataloader:", dloader)
    if stage in ("fit", "val"):
        dloader = datamodule.val_dataloader()
        print("val dataloader:", dloader)
    if stage == "test":
        dloader = datamodule.test_dataloader()
        print("test dataloader:", dloader)
    if stage in ("submission", "test-on-submission"):
        dloader = datamodule.submission_dataloader()
        print("submission dataloader:", dloader)

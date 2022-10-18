from pathlib import Path
from typing import Dict, Optional

import openpack_toolkit as optk
import pytest
from omegaconf import OmegaConf
from openpack_torch.data.datamodule import OpenPackBaseDataModule
from openpack_torch.data.datasets import OpenPackImu as OpenPackImuDataset


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
            "stream": optk.configs.datasets.streams.ATR_ACC_STREAM,
            "annotation": optk.configs.datasets.annotations.ACTIVITY_1S_ANNOTATION,
            "split": {
                "name": "unit-test",
                "train": [("U0102", "S0500")],
                "val": [("U0102", "S0500")],
                "test": [("U0102", "S0500")],
                "submission": [("U0102", "S0500")],
            },
        },
        "train": {
            "window": 30 * 60,
            "batch_size": 32,
            "num_workers": 3,
            "debug": {
                "batch_size": 2,
            },
        },
        "debug": False,
    })
    return cfg

# ------------------------------------------------------------------------------


@pytest.mark.parametrize("stage, debug", (
    (None, True),
    (None, False),
    ("fit", False),
    ("validate", False),
    ("test", False),
    ("submission", False),
    ("test-on-submission", False),
))
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

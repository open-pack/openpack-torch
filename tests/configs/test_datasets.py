import openpack_toolkit as optk
import pytest
from omegaconf import OmegaConf
from openpack_torch.configs import datasets


@pytest.mark.parametrize("conf", (
    datasets.OPENPACK_ACC_DATASET_CONFIG,
    datasets.OPENPACK_2D_KEYPOINT_DATASET_CONFIG,
))
def test_DatasetConfig__01(conf):
    print(OmegaConf.to_yaml(conf))


def test_DatasetConfig__02():
    """ Replace data splits """
    conf = datasets.OPENPACK_ACC_DATASET_CONFIG
    conf.split = optk.configs.datasets.splits.DEBUG_SPLIT
    print(OmegaConf.to_yaml(conf))

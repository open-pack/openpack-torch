import openpack_toolkit as optk
from hydra.core.config_store import ConfigStore

from .datasets import (OPENPACK_2D_KEYPOINT_DATASET_CONFIG,
                       OPENPACK_ACC_DATASET_CONFIG)


def register_configs() -> None:
    cs = ConfigStore.instance()

    data = {
        "user": [
            optk.configs.users.U0102,
            optk.configs.users.U0103,
            optk.configs.users.U0105,
            optk.configs.users.U0106,
            optk.configs.users.U0107,
        ],
        "dataset/stream": [
            optk.configs.datasets.streams.ATR_ACC_STREAM,
            optk.configs.datasets.streams.ATR_QAGS_STREAM,
            optk.configs.datasets.streams.KINECT_2D_KPT_STREAM,
        ],
        "dataset/split": [
            optk.configs.datasets.splits.DEBUG_SPLIT,
            optk.configs.datasets.splits.PILOT_CHALLENGE_SPLIT,
        ],
        "dataset/annotation": [
            optk.configs.datasets.annotations.OPENPACK_ACTIONS_ANNOTATION,
            optk.configs.datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
        ],
        "dataset": [
            OPENPACK_ACC_DATASET_CONFIG, OPENPACK_2D_KEYPOINT_DATASET_CONFIG
        ],
    }
    for group, items in data.items():
        for item in items:
            cs.store(group=group, name=item.name, node=item)

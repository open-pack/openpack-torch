import openpack_toolkit as optk
from hydra.core.config_store import ConfigStore


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
            optk.configs.datasets.streams.OPENPACK_ATR_ACC_STREAM,
            optk.configs.datasets.streams.OPENPACK_ATR_QAGS_STREAM,
            optk.configs.datasets.streams.OPENPACK_KINECT_2D_KPT_STREAM,
        ],
        "dataset/split": [
            optk.configs.datasets.splits.DEBUG_SPLIT,
            optk.configs.datasets.splits.PILOT_CHALLENGE_SPLIT,
        ],
        "dataset/annotation": [
            optk.configs.datasets.annotations.OPENPACK_ACTIONS_ANNOTATION,
            optk.configs.datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
        ],
    }
    for group, items in data.items():
        for item in items:
            cs.store(group=group, name=item.name, node=item)

from openpack_toolkit.configs import DatasetConfig, datasets

OPENPACK_ACC_DATASET_CONFIG = DatasetConfig(
    name="openpack-acc",
    streams={
        "atr-acc": datasets.streams.ATR_ACC_STREAM,
    },
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annot=datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
)

OPENPACK_2D_KEYPOINT_DATASET_CONFIG = DatasetConfig(
    name="openpack-2d-kpt",
    streams={
        "kinect-2d-kpt": datasets.streams.KINECT_2D_KPT_STREAM
    },
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annot=datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
)

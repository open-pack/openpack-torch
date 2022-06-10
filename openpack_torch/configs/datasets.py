from openpack_toolkit.configs import DatasetConfig, datasets

OPENPACK_ACC_DATASET_CONFIG = DatasetConfig(
    name="openpack-acc",
    streams=[datasets.streams.OPENPACK_ATR_ACC_STREAM],
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annot=datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
)

OPENPACK_2D_KEYPOINT_DATASET_CONFIG = DatasetConfig(
    name="openpack-2d-kpt",
    streams=[datasets.streams.OPENPACK_KINECT_2D_KPT_STREAM],
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annot=datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
)

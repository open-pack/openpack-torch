from openpack_toolkit.configs import DatasetConfig, datasets
from openpack_toolkit import ActSet

OPENPACK_ACC_DATASET_CONFIG = DatasetConfig(
    name="openpack-acc",
    streams=None,
    stream=datasets.streams.ATR_ACC_STREAM,
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annotation=datasets.annotations.OPENPACK_OPERATIONS_1HZ_ANNOTATION,
    classes=datasets.annotations.OPENPACK_OPERATIONS,
)

OPENPACK_2D_KEYPOINT_DATASET_CONFIG = DatasetConfig(
    name="openpack-2d-kpt",
    streams=None,
    stream=datasets.streams.KINECT_2D_KPT_STREAM,
    split=datasets.splits.PILOT_CHALLENGE_SPLIT,
    annotation=datasets.annotations.OPENPACK_OPERATIONS_1HZ_ANNOTATION,
    classes=datasets.annotations.OPENPACK_OPERATIONS,
)

from logging import getLogger
from pathlib import Path

import numpy as np
import openpack_toolkit as optk
import pandas as pd
from omegaconf import DictConfig, open_dict
from openpack_toolkit import ActSet

from ._baseclass import Sequence

logger = getLogger(__name__)
# -----------------------------------------------------------------------------

ATR_ATTACHED_IN_WRONG_DIRECTION = (
    "U0103-S0300-atr03",
    "U0103-S0400-atr03",
    "U0107-S0400-atr03",
    "U0201-S0500-atr02",
    "U0204-S0100-atr02",
    "U0204-S0500-atr02",
    "U0207-S0100-atr02",
    "U0207-S0200-atr03",
    "U0209-S0400-atr03",
    "U0209-S0500-atr03",
)


def load_imu_wrapper(cfg: DictConfig) -> Sequence:
    has_spec = hasattr(cfg.dataset.stream, "spec")

    paths_imu = []
    devices = (
        cfg.dataset.stream.spec.imu.devices
        if has_spec
        else cfg.dataset.stream.imu.devices
    )
    for device in devices:
        with open_dict(cfg):
            cfg.device = device

        if has_spec:
            path = Path(
                cfg.dataset.stream.spec.imu.path.dir,
                cfg.dataset.stream.spec.imu.path.fname,
            )
        else:
            path = Path(
                cfg.dataset.stream.imu.path.dir,
                cfg.dataset.stream.imu.path.fname,
            )
        paths_imu.append(path)

    ts_sess, x_sess = optk.data.load_imu(
        paths_imu,
        use_acc=cfg.dataset.stream.spec.imu.acc,
        use_gyro=cfg.dataset.stream.spec.imu.gyro,
        use_quat=cfg.dataset.stream.spec.imu.quat,
    )

    # Fix ATR data attached in the wrong directions
    # Multiply -1 to fix the direction
    if cfg.dataset.stream.spec.imu.acc:
        # assert cfg.dataset.stream.spec.imu.gyro is False
        # assert cfg.dataset.stream.spec.imu.quat is False

        for i, device in enumerate(cfg.dataset.stream.spec.imu.devices):
            key = f"{cfg.user.name}-{cfg.session}-{device}"
            if key in ATR_ATTACHED_IN_WRONG_DIRECTION:
                x_sess[i * 3 + 0] *= -1  # x-axis
                x_sess[i * 3 + 1] *= -1  # y-axis
                logger.warning(
                    f"Acc data of {key}[ind={i*3}, {i*3+1}] is fix by multiply -1. "
                    f"x_sess={x_sess.shape}"
                )

    seq = Sequence(ts_sess, x_sess, metadata={"type": "imu"})
    return seq


# -----------------------------------------------------------------------------


def load_iot_data_wrapper(cfg: DictConfig, ts_sess: np.ndarray) -> Sequence:
    path_ht = Path(
        cfg.dataset.stream.spec.iot.spec.devices.spec.ht.spec.path.dir,
        cfg.dataset.stream.spec.iot.spec.devices.spec.ht.spec.path.fname,
    )

    path_printer = Path(
        cfg.dataset.stream.spec.iot.spec.devices.spec.printer.spec.path.dir,
        cfg.dataset.stream.spec.iot.spec.devices.spec.printer.spec.path.fname,
    )

    anchor_params = cfg.dataset.stream.spec.iot.spec.anchor
    if anchor_params.dim == 2:
        ht_sess = optk.data.load_and_resample_scan_log(path_ht, ts_sess)
        printer_sess = optk.data.load_and_resample_scan_log(path_printer, ts_sess)

        anchor_sess = np.stack([ht_sess, printer_sess], axis=0)

    elif anchor_params.dim == 1:
        raise NotImplementedError("Integrated anchor is disabled.")

    seq = Sequence(ts_sess, anchor_sess, metadata={"type": "IoT"})
    return seq


# -----------------------------------------------------------------------------


def load_annot_wrapper(
    cfg: DictConfig, ts_sess: np.ndarray, submission: bool, classes: optk.ActSet
) -> Sequence:
    if submission:
        # For set dummy data.
        label = np.zeros((len(ts_sess),), dtype=np.int64)
    else:
        path = Path(
            cfg.dataset.annotation.spec.path.dir, cfg.dataset.annotation.spec.path.fname
        )

        if cfg.dataset.annotation.metadata.labels.type == "annotation/operation":
            # df_label = optk.data.load_and_resample_annotation(
            #     path, ts_sess, classes=classes)
            df_label = load_and_resample_annotation(path, ts_sess, classes=classes)
            # df_label["act_id"].replace(8200, 8100)
        elif cfg.dataset.annotation.metadata.labels.type == "annotation/action":
            assert cfg.dataset.annotation.metadata.labels.label_format == "soft-target"
            df_label = load_annotation_action(path, ts_sess, classes=classes)

        label = df_label["act_idx"].values

    seq = Sequence(ts_sess, label[np.newaxis], metadata={"type": "annot/operation"})
    return seq


def load_and_resample_annotation(
    path: Path,
    unixtimes_ms: np.ndarray,
    classes: ActSet,
    label_col: str = "id",
) -> pd.DataFrame:
    """Load annotation data and resample them according to unixtime sequence ``T``.
    If there are no annotation records for the given timestamp, that records is treated
    as NULL class.
    Args:
        path (Path): path to annotation CSV file.
        unixitmes (np.ndarray): unixtime seqeuence (milli-scond precision).
    Returns:
        pd.DataFrame: -
    """
    null_class_id = classes.get_ignore_class_id()
    if isinstance(null_class_id, tuple):
        null_class_id = null_class_id[-1]

    df = pd.read_csv(path)
    logger.debug(f"load annotation data from {path} -> df={df.shape}")
    ut_min, ut_max = df["unixtime"].min(), df["unixtime"].max()

    null_record = df.head(1).copy()
    null_record["unixtime"] = 0
    null_record["box"] = 0
    null_record[label_col] = null_class_id
    df = pd.concat([df, null_record], axis=0, ignore_index=True)

    # unixtime with second precision.
    unixtimes_sec = unixtimes_ms - (unixtimes_ms % 1000)
    # Assing 0 to non-annotated sequence.
    unixtimes_sec[unixtimes_sec < ut_min] = 0
    unixtimes_sec[unixtimes_sec > ut_max] = 0

    df = df.rename(columns={"unixtime": "annot_time"}).set_index("annot_time")
    df = df.loc[unixtimes_sec, :].reset_index(drop=False)
    df["unixtime"] = unixtimes_ms

    df["act_id"] = df[label_col]
    # DEBUG: Replace 8200 with 8100
    df["act_id"].replace(8200, 8100, inplace=True)

    df["act_idx"] = classes.convert_id_to_index(df["act_id"].values)

    cols = ["unixtime", "annot_time", "user", "session", "box", "act_id", "act_idx"]
    return df[cols]


def load_annotation_action(
    path: Path,
    unixtimes_ms: np.ndarray,
    classes: ActSet,
    label_col: str = None,
) -> pd.DataFrame:
    """Load annotation data and resample them according to unixtime sequence ``T``.
    If there are no annotation records for the given timestamp, that records is treated
    as NULL class.
    Args:
        path (Path): path to annotation CSV file.
        unixitmes (np.ndarray): unixtime seqeuence (milli-scond precision).
    Returns:
        pd.DataFrame: -
    """
    action_cols = [f"ID{cls.id}" for cls in classes.classes if cls.id != 8106]

    exclude_class_id = classes.classes[-1].id
    assert exclude_class_id == 8106, exclude_class_id

    df = pd.read_csv(path)
    logger.debug(f"load annotation data from {path} -> df={df.shape}")
    ut_min, ut_max = df["unixtime"].min(), df["unixtime"].max()

    null_record = df.head(1).copy()
    null_record["unixtime"] = 0
    null_record["box"] = 0
    null_record[action_cols] = 0.0
    null_record[exclude_class_id] = 1.0
    df = pd.concat([df, null_record], axis=0, ignore_index=True)

    # unixtime with second precision.
    unixtimes_sec = unixtimes_ms - (unixtimes_ms % 1000)
    # Assing 0 to non-annotated sequence.
    unixtimes_sec[unixtimes_sec < ut_min] = 0
    unixtimes_sec[unixtimes_sec > ut_max] = 0

    df = df.rename(columns={"unixtime": "annot_time"}).set_index("annot_time")
    df = df.loc[unixtimes_sec, :].reset_index(drop=False)
    df["unixtime"] = unixtimes_ms

    df["act_idx"] = np.argmax(df[action_cols].values, axis=1)
    action_ids = {i: cls.id for i, cls in enumerate(classes.classes)}
    df["act_id"] = df["act_idx"].apply(lambda cls_idx: action_ids[cls_idx])

    cols = ["unixtime", "annot_time", "user", "session", "box", "act_id", "act_idx"]
    return df[cols]


def load_annot_action_wrapper(
    cfg: DictConfig, ts_sess: np.ndarray, submission: bool, classes: optk.ActSet
) -> Sequence:
    if submission:
        # For set dummy data.
        label = np.zeros((len(ts_sess),), dtype=np.int64)
    else:
        # TODO: Update schema.
        path = Path(
            cfg.dataset.annotation.path.dir,
            cfg.dataset.annotation.path.fname,
        )
        df_label = load_annotation_action(path, ts_sess, classes=classes)
        label = df_label["act_idx"].values

    seq = Sequence(ts_sess, label[np.newaxis], metadata={"type": "annot/action"})
    return seq


# -----------------------------------------------------------------------------


def load_kinect_2d_kpt_wrapper(cfg: DictConfig) -> Sequence:
    path = Path(
        cfg.dataset.stream.spec.kinect2dKpt.spec.path.dir,
        cfg.dataset.stream.spec.kinect2dKpt.spec.path.fname,
    )
    ts_sess, x_sess = optk.data.load_keypoints(path)

    seq = Sequence(ts_sess, x_sess, metadata={"type": "kinect/2dKpt"})
    return seq

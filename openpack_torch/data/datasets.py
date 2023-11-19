"""Dataset Class for OpenPack dataset.
"""
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np
import openpack_toolkit as optk
import torch
from omegaconf import DictConfig, open_dict

from ._baseclass import Sequence, SequenceSet, Window
from ._wrapper import (
    load_annot_wrapper,
    load_imu_wrapper,
    load_iot_data_wrapper,
    load_kinect_2d_kpt_wrapper,
)
from .preprocessing import compute_semantic_hard_boundary

log = getLogger(__name__)


def random_window_shift(win: Window, win_size: int, seq_len: int) -> Window:
    """Change cropping position up to 50%.

    Args:
        seq_len (int): length of whole sequence.
    """
    start = win.start
    stop = start + win_size
    if stop >= seq_len:
        stop = seq_len

    shift = int(np.random.uniform(-1, 1, size=(1,)) * ((stop - start) / 2))
    new_start, new_stop = start + shift, stop + shift

    if new_start < 0:
        new_start = 0
    if new_stop >= seq_len:
        new_stop = seq_len

    if new_start > new_stop:
        raise ValueError(
            f"start={start}, stop={stop} "
            f"-> new_start={new_start}, new_stop={new_stop} (shift={shift})"
        )

    new_win = Window(win.sequence_idx, win.segment_idx, new_start, new_stop)
    return new_win


# -----------------------------------------------------------------------------


class OpenPackImu(torch.utils.data.Dataset):
    """Dataset class for IMU data.
    Attributes:
        data (List[Dict]): each sequence is stored in dict. The dict has 5 keys (i.e.,
            user, session, data, label(=class index), unixtime). data is a np.ndarray with
            shape = ``(N, channel(=acc_x, acc_y, ...), window, 1)``.
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (optk.ActSet): list of activity classes.
        window (int): sliding window size.
        debug (bool): If True, enable debug mode. Default to False.
        submission (bool): Set True when you make submission file. Annotation data will not be
            loaded and dummy data will be generated. Default to False.
    Todo:
        * Make a minimum copy of cfg (DictConfig) before using in ``load_dataset()``.
        * Add method for parameter validation (i.e., assert).
    """

    data: List[Dict] = None
    index: Tuple[Dict] = None
    tensor_set_config: Dict = {
        "data": {
            "imu": {
                "new_key": "x",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
        },
        "labels": {
            "annot": {
                "new_key": "t",
                "dtype": torch.long,
                "squeeze": True,
                "callbacks": [
                    lambda x: x.squeeze(0),
                ],
            },
        },
        "unixtime": {
            "new_key": "ts",
            "dtype": torch.long,
        },
    }

    def __init__(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...] = None,
        classes: optk.ActSet = None,
        window: int = 30 * 60,
        random_crop=False,
        submission: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize OpenPackImu dataset class.
        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.cfg = cfg
        self.classes = classes
        self.window = window
        self.submission = submission
        self.debug = debug
        self.random_crop = random_crop

        if self.classes is None:
            class_set_key = cfg.dataset.annotation.name.replace("-", "_").upper()
            classes_tuple = eval(
                f"optk.configs.datasets.annotations.{class_set_key}"
            ).classes
            self.classes = optk.ActSet(classes_tuple)

        if user_session_list is not None:
            self.load_dataset(cfg, user_session_list, window, submission=submission)
            self.preprocessing()

    def load_single_session(self, cfg, submission) -> SequenceSet:
        data_seq = dict()
        data_seq["imu"] = load_imu_wrapper(cfg)
        base_unixtime_seq = data_seq["imu"].unixtime

        labels_seq = dict()
        labels_seq["annot"] = load_annot_wrapper(
            cfg, base_unixtime_seq, submission, self.classes
        )

        ss = SequenceSet(
            user=cfg.user.name,
            session=cfg.session,
            data=data_seq,
            labels=labels_seq,
            primary_seqence="imu",
        )
        return ss

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        window: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.
        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session_list):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session
            ss = self.load_single_session(cfg, submission)

            data.append(ss)
            index += [
                Window(seq_idx, seg_idx, start, start + window)
                for seg_idx, start in enumerate(range(0, ss.seq_len(), window))
            ]

        self.data = data
        self.index = tuple(index)

    def preprocessing(self) -> None:
        """
        * Normalize [-3G, +3G] into [0, 1].
        """
        # NOTE: Normalize ACC data. ([-3G, +3G] -> [0, 1])
        # NOTE: Described in Appendix Sec.3.2.
        is_agq = (
            self.cfg.dataset.stream.spec.imu.gyro
            and self.cfg.dataset.stream.spec.imu.quat
        )
        devices = self.cfg.dataset.stream.spec.imu.devices

        for ss in self.data:
            x = ss.data.get("imu").data
            if is_agq:
                for i, device in enumerate(devices):
                    # ACC
                    dims = [10 * i + 0, 10 * i + 1, 10 * i + 2]
                    x[dims] = np.clip(x[dims], -3, +3)
                    x[dims] = (x[dims] + 3.0) / 6.0
                    # Gyro
                    for j, ch in enumerate(["x", "y", "z"]):
                        dim = 10 * i + (3 + j)
                        mean = (
                            self.cfg.dataset.stream.spec.imu.stats[device].gyro[ch].mean
                        )
                        std = (
                            self.cfg.dataset.stream.spec.imu.stats[device].gyro[ch].std
                        )
                        x[dim] = (x[dim] - mean) / std
            else:
                x = np.clip(x, -3, +3)
                x = (x + 3.0) / 6.0
            ss.data.get("imu").data = x

    @property
    def num_classes(self) -> int:
        """Returns the number of classes
        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImu("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}, "
            f"random_crop={self.random_crop}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        win = self.index[index]
        ss = self.data[win.sequence_idx]

        # TODO: Implement  Random crop
        if self.random_crop:
            win = random_window_shift(win, self.window, len(ss))

        new_ss = ss.get_segment(win, self.window)
        tensors = new_ss.get_tensors(self.tensor_set_config)
        return tensors


# -----------------------------------------------------------------------------
# =====================
#   With Boundary Info
# =====================
class ImuBoundaryDataset(OpenPackImu):
    tensor_set_config: Dict = {
        "data": {
            "imu": {
                "new_key": "x",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
        },
        "labels": {
            "annot": {
                "new_key": "t",
                "dtype": torch.long,
                "callbacks": [
                    lambda x: x.squeeze(0),
                ],
            },
            "boundary": {
                "new_key": "tb",
                "dtype": torch.float,
            },
        },
        "unixtime": {
            "new_key": "ts",
            "dtype": torch.long,
        },
    }

    def preprocessing(self) -> None:
        """
        * Compute boundary label.
        """
        super().preprocessing()

        # -- Add action boundary labels --
        for ss in self.data:
            t_id = ss.labels["annot"].data[0]
            unixtime = ss.labels["annot"].unixtime
            metadata = {"type": "boundary"}

            t_bd = compute_semantic_hard_boundary(t_id, len(self.classes))
            ss.labels["boundary"] = Sequence(unixtime, t_bd, metadata)


# =================
#   With IoT Data
# =================
class ImuIoTDataset(OpenPackImu):
    tensor_set_config: Dict = {
        "data": {
            "imu": {
                "new_key": "x",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
            "iot": {
                "new_key": "x_iot",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
        },
        "labels": {
            "annot": {
                "new_key": "t",
                "dtype": torch.long,
                # "squeeze": True,
                "callbacks": [
                    lambda x: x.squeeze(0),
                ],
            },
        },
        "unixtime": {
            "new_key": "ts",
            "dtype": torch.long,
        },
    }

    def load_single_session(self, cfg, submission) -> SequenceSet:
        data_seq = dict()
        data_seq["imu"] = load_imu_wrapper(cfg)
        base_unixtime_seq = data_seq["imu"].unixtime

        data_seq["iot"] = load_iot_data_wrapper(cfg, base_unixtime_seq)

        labels_seq = dict()
        labels_seq["annot"] = load_annot_wrapper(
            cfg, base_unixtime_seq, submission, self.classes
        )

        ss = SequenceSet(
            user=cfg.user.name,
            session=cfg.session,
            data=data_seq,
            labels=labels_seq,
            primary_seqence="imu",
            # metadata=None,
        )
        return ss


class ImuBoundaryIoTDataset(ImuIoTDataset):
    tensor_set_config: Dict = {
        "data": {
            "imu": {
                "new_key": "x",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
            "iot": {
                "new_key": "x_iot",
                "dtype": torch.float,
                "callbacks": [
                    lambda x: x.unsqueeze(2),
                ],
            },
        },
        "labels": {
            "annot": {
                "new_key": "t",
                "dtype": torch.long,
                "callbacks": [
                    lambda x: x.squeeze(0),
                ],
            },
            "boundary": {
                "new_key": "tb",
                "dtype": torch.float,
            },
        },
        "unixtime": {
            "new_key": "ts",
            "dtype": torch.long,
        },
    }

    def preprocessing(self) -> None:
        """
        * Compute boundary label.
        """
        super().preprocessing()

        # -- Add action boundary labels --
        for ss in self.data:
            t_id = ss.labels["annot"].data[0]
            unixtime = ss.labels["annot"].unixtime
            metadata = {"type": "boundary"}

            t_bd = compute_semantic_hard_boundary(t_id, len(self.classes))
            ss.labels["boundary"] = Sequence(unixtime, t_bd, metadata)


# =============
#   Keypoints
# =============
class Kinect2dKptDataset(OpenPackImu):
    tensor_set_config: Dict = {
        "data": {
            "kinect2dKpt": {
                "new_key": "x",
                "dtype": torch.float,
            },
        },
        "labels": {
            "annot": {
                "new_key": "t",
                "dtype": torch.long,
                "callbacks": [
                    lambda x: x.squeeze(0),
                ],
            },
        },
        "unixtime": {
            "new_key": "ts",
            "dtype": torch.long,
        },
    }

    def load_single_session(self, cfg, submission) -> SequenceSet:
        data_seq = dict()
        data_seq["kinect2dKpt"] = load_kinect_2d_kpt_wrapper(cfg)
        base_unixtime_seq = data_seq["kinect2dKpt"].unixtime

        labels_seq = dict()
        labels_seq["annot"] = load_annot_wrapper(
            cfg, base_unixtime_seq, submission, self.classes
        )

        ss = SequenceSet(
            user=cfg.user.name,
            session=cfg.session,
            data=data_seq,
            labels=labels_seq,
            primary_seqence="kinect2dKpt",
        )
        return ss

    def preprocessing(self) -> None:
        """Standardize X-/Y-axis."""
        mean_x = self.cfg.dataset.stream.spec.kinect2dKpt.spec.stats.mean.x
        mean_y = self.cfg.dataset.stream.spec.kinect2dKpt.spec.stats.mean.y
        std_x = self.cfg.dataset.stream.spec.kinect2dKpt.spec.stats.std.x
        std_y = self.cfg.dataset.stream.spec.kinect2dKpt.spec.stats.std.y

        for ss in self.data:
            x = ss.data.get("kinect2dKpt").data
            x[0] = (x[0] - mean_x) / std_x
            x[1] = (x[1] - mean_y) / std_y
            ss.data.get("kinect2dKpt").data = x

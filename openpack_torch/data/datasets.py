"""Dataset Class for OpenPack dataset.
"""
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openpack_toolkit as optk
import torch
from openpack_toolkit import OPENPACK_OPERATIONS

logger = getLogger(__name__)


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
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            rootdir: Path,
            user_session: Tuple[Tuple[int, int], ...],
            imu_nodes: Tuple[str] = None,
            use_acc: bool = True,
            use_gyro: bool = False,
            use_quat: bool = False,
            debug: bool = False,
            window: int = 30 * 60,
            submission: bool = False,
            classes: optk.ActSet = OPENPACK_OPERATIONS,
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            rootdir (Path): path to the rootdirectory of OpenPack dataset.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            imu_nodes (Tuple[str], optional): the list of IMU node names. Defaults to None.
            use_acc (bool, optional): If True, include acceleration data. Defaults to True.
            use_gyro (bool, optional): If True, include gyroscope data. Defaults to False.
            use_quat (bool, optional): If True, include quaternion data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
            window (int, optional): window size [steps]. Defaults to 30*60.
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
        """
        super().__init__()
        self.window = window
        self.debug = debug
        self.submission = submission
        self.classes = classes

        if imu_nodes is None:
            imu_nodes = ("atr01", "atr02", "atr03", "atr04")
        self.load_dataset(
            rootdir,
            user_session,
            imu_nodes,
            window,
            use_acc=use_acc,
            use_gyro=use_gyro,
            use_quat=use_quat,
            submission=submission,
        )
        self.preprocessing()

    def load_dataset(
        self,
        rootdir: Path,
        user_session: Tuple[Tuple[int, int], ...],
        imu_nodes: Tuple[str, ...],
        window: int = None,
        use_acc: bool = True,
        use_gyro: bool = False,
        use_quat: bool = False,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            rootdir (Path): _description_
            user_session (Tuple[Tuple[int, int], ...]): _description_
            imu_nodes (Tuple[str, ...]): _description_
            window (int, optional): _description_. Defaults to None.
            use_acc (bool, optional): _description_. Defaults to True.
            use_gyro (bool, optional): _description_. Defaults to False.
            use_quat (bool, optional): _description_. Defaults to False.
            submission (bool, optional): _description_. Defaults to False.
        """
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session):
            paths_imu = tuple([Path(
                rootdir, f"U{user:0=4}", f"{node}", f"S{session:0=4}.csv",
            ) for node in imu_nodes])
            ts_sess, x_sess = optk.data.load_imu(
                paths_imu, use_acc=use_acc, use_gyro=use_gyro, use_quat=use_quat)

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path_annot = Path(
                    rootdir,
                    f"U{user:0=4}",
                    "annotation/work-process",
                    f"S{session:0=4}.csv",
                )
                df_label = optk.data.load_annotation(
                    path_annot, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_sess,
                "label": label,
                "unixtime": ts_sess,
            })

            seq_len = ts_sess.shape[0]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, window))]
        self.data = data
        self.index = tuple(index)

    def preprocessing(self) -> None:
        """This method is called after ``load_dataset()`` and apply preprocessing to loaded data.
        """
        logger.warning("No preprocessing is applied.")

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
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail, np.newaxis]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}

# -----------------------------------------------------------------------------


class OpenPackKeypoint(torch.utils.data.Dataset):
    """Dataset Class for Keypoint Data.

    Attributes:
        data (List[Dict]): shape = (N, 3, FRAMES, VERTEX)
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (Tuple[ActClass]): list of activity classes.
        window (int): window size (=the number of frames per sample)
        device (torch.device): -
        dtype (Tuple[torch.dtype,torch.dtype]): -
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            rootdir: Path,
            user_session: Tuple[Tuple[int, int], ...],
            keypoint_type: str = None,
            debug: bool = False,
            window: int = 15 * 60,
            submission: bool = False,
            classes: optk.ActSet = OPENPACK_OPERATIONS,
    ) -> None:
        """Initialize OpenPackKyepoint dataset class.

        Args:
            rootdir (Path): _description_
            keypoint_type (str): _description_
            user_session (Tuple[Tuple[int, int], ...]): _description_
            debug (bool, optional): enable debug mode. Defaults to False.
            window (int, optional): window size. Defaults to 15*60 [frames].
            submission (bool, optional): _description_. Defaults to False.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
        """
        super().__init__()
        self.window = window
        self.debug = debug
        self.classes = classes
        self.submission = submission

        self.load_dataset(
            rootdir,
            keypoint_type,
            user_session,
            submission=submission)
        self.preprocessing()

    def load_dataset(
        self,
        rootdir: Path,
        keypoint_type: str,
        user_session: Tuple[Tuple[int, int], ...],
        submission: bool = False,
    ):
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session):
            path_skeleton = Path(
                rootdir,
                f"U{user:0=4}",
                "pose/",
                keypoint_type,
                f"S{session:0=4}.json",
            )
            ts_sess, x_sess = optk.data.load_keypoints(path_skeleton)
            x_sess = x_sess[:(x_sess.shape[0] - 1)]  # Remove prediction score.

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path_annot = Path(
                    rootdir,
                    f"U{user:0=4}",
                    "annotation/work-process",
                    f"S{session:0=4}.csv",
                )
                df_label = optk.data.load_annotation(
                    path_annot, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_sess,
                "label": label,
                "unixtime": ts_sess,
            })

            seq_len = x_sess.shape[1]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, self.window))]

        self.data = data
        self.index = tuple(index)

    def preprocessing(self):
        """This method is called after ``load_dataset()`` method and apply preprocessing to loaded data.

        Todo:
            - [ ] sklearn.preprocessing.StandardScaler()
            - [ ] DA (half_body_transform)
                - https://github.com/open-mmlab/mmskeleton/blob/b4c076baa9e02e69b5876c49fa7c509866d902c7/mmskeleton/datasets/estimation.py#L62
        """
        logger.warning("No preprocessing is applied.")

    @ property
    def num_classes(self) -> int:
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackKeypoint("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        # TODO: Make utilities to extract window from long sequence.
        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}

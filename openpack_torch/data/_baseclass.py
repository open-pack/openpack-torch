import copy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class Sequence:
    def __init__(
        self,
        unixtime: np.ndarray,
        data: np.ndarray,
        metadata: Dict = None,
    ):
        self.unixtime = unixtime
        self.data = data

        self.metadata = metadata
        if self.metadata is None:
            self.metadata = dict()

        assert self.unixtime.dtype == np.int64, (
            f"unixtime must be np.int64, but got {self.unixtime.dtype}"
        )

        assert data.ndim >= 2, (
            f"data.ndim should be >= 2, but got data.shape={data.shape}"
            f" and dim=0/1 should be ch-/time-axis.: {metadata}"
        )
        assert len(self.unixtime) == data.shape[1], (
            "the length of timestanp and data array is not equal. "
            f" unixtime={len(self.unixtime)}, data={self.data.shape[1]}"
        )

    def __len__(self) -> int:
        return len(self.unixtime)

    def __str__(self) -> str:
        return (
            "Sequence("
            f"unixtime={self.unixtime.shape},"
            f"data={self.data.shape}"
            # f"metadata={self.metadata}"
            ")"
        )

    def __getitem__(self, key):
        """
        """
        metadata = copy.deepcopy(self.metadata)
        metadata.update({
            "slice": key,
        })
        return Sequence(
            copy.deepcopy(self.unixtime[key]),
            copy.deepcopy(self.data[:, key]),
            metadata,
        )

    def pad_tail(self, pad_size: int):
        assert isinstance(pad_size, int)
        assert pad_size > 0

        self.unixtime = np.pad(
            self.unixtime,
            [(0, pad_size)],
            mode="constant",
            constant_values=self.unixtime[-1],
        )

        pad = [(0, 0), (0, pad_size)]
        if self.data.ndim > 2:
            pad += [(0, 0)] * (self.data.ndim - 2)
        self.data = np.pad(
            self.data,
            pad,
            mode="constant",
            constant_values=0,
        )


@dataclass
class Window:
    def __init__(
            self,
            sequence_idx: int,
            segment_idx: int,
            start: int,
            stop: int,
    ):
        self.sequence_idx = sequence_idx
        self.segment_idx = segment_idx
        self.start = start
        self.stop = stop

        assert start >= 0, f"start shall be positive, but got {start}"
        assert stop > start, f"end shall be larger than start, but got {stop}"

    def __len__(self):
        return self.stop - self.start

    def __repr__(self):
        return (
            "Win("
            f"seq={self.sequence_idx},"
            f"seg={self.segment_idx},"
            f"start={self.start},stop={self.stop}"
            ")"
        )

    def __eq__(self, other):
        # checking both objects of same class
        if other is None or not isinstance(self, type(other)):
            return False
        return self.__dict__ == other.__dict__

    def get_slice(self):
        return slice(self.start, self.stop, None)


@dataclass
class SequenceSet:
    def __init__(
            self,
            user: str = None,
            session: str = None,
            data: Dict[str, Sequence] = None,
            labels: Dict[str, Sequence] = None,
            primary_seqence: str = None,
            metadata: Dict = None,
    ):
        self.user: str = user
        self.session: str = session
        self.metadata = metadata
        self.data: Dict[str, Sequence] = data
        self.labels: Dict[str, Sequence] = labels
        self.primary_seqence = primary_seqence
        if self.primary_seqence is None:
            self.primary_seqence = self.data.keys()[0]

    def __len__(self):
        return len(self.data[self.primary_seqence])

    def __str__(self):
        return (
            "SequenceSet("
            f"user={self.user},session={self.session},"
            f"data={list(self.data.keys())},"
            f"labels={list(self.labels.keys())},"
            f"metadata={self.metadata}"
        )

    def get_primary_unixtime(self):
        return self.data[self.primary_seqence].unixtime

    def seq_len(self) -> int:
        """Returns the sequence length of the primary sequence.

        Returns:
            int
        """
        return len(self.data[self.primary_seqence])

    def __getitem__(self, slice_key):
        new_data = dict()
        for key in self.data.keys():
            new_data[key] = self.data[key][slice_key]

        new_labels = dict()
        for key in self.labels.keys():
            new_labels[key] = self.labels[key][slice_key]

        new_ss = SequenceSet(
            user=self.user,
            session=self.session,
            data=new_data,
            labels=new_labels,
            primary_seqence=self.primary_seqence,
            metadata=copy.deepcopy(self.metadata),
        )
        return new_ss

    def get_segment(self, win: Window, win_size: int) -> Sequence:
        new_ss = self[win.get_slice()]

        # Padding
        if len(new_ss) < win_size:
            pad_tail = win_size - len(new_ss)
            for key in new_ss.data.keys():
                new_ss.data[key].pad_tail(pad_tail)
            for key in new_ss.labels.keys():
                new_ss.labels[key].pad_tail(pad_tail)
        return new_ss

    def get_tensors(self, target: Dict) -> Dict:
        """Retuns dict of tensors.

        Args:
            target (Dict): dict of target key and datatype

        Returns:
            Dict
        """
        def _to_torch(arr, conf: Dict):
            x = torch.from_numpy(arr)
            if "dtype" in conf.keys():
                x = x.to(dtype=conf["dtype"])
            for clbk_func in conf.get("callbacks", []):
                x = clbk_func(x)
            return x

        tensors = dict()

        # data
        for key, d in target["data"].items():
            new_key = d.get("new_key", key)
            tensors[new_key] = _to_torch(self.data[key].data, d)
        # labels
        for key, d in target["labels"].items():
            new_key = d.get("new_key", key)
            tensors[new_key] = _to_torch(self.labels[key].data, d)
        # unixtime
        if "unixtime" in target.keys():
            new_key = target["unixtime"].get("new_key", key)
            d = target["unixtime"]
            tensors[new_key] = _to_torch(
                self.data[self.primary_seqence].unixtime, d)

        return tensors
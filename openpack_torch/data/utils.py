import copy
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import openpack_torch as optorch

log = getLogger(__name__)


def assemble_sequence_list_data_volume_flexible_cv(cfg: DictConfig, stage: str):
    src_set = cfg.metadata.labels.src_set

    if stage in ("fit", "train"):
        seq_list = cfg.dataset.split.spec.pool[src_set].train
    elif stage == "test":
        # Predict on all the other sessions.
        seq_list = []
        for key, d in cfg.dataset.split.spec.pool.items():
            seq_list += list(d.train)
            seq_list += list(d.test)
    elif stage == "test-b2":
        # Predict on all the other sessions.
        seq_list = cfg.dataset.split.spec.pool[src_set].test
    elif stage == "test-b3":
        # Predict on all the other sessions.
        seq_list = []
        exclude_list = [
            f"{user}-{sess}"
            for user, sess in cfg.dataset.split.spec.pool[src_set].get(
                "exclude_on_test", []
            )
        ]
        for key, d in cfg.dataset.split.spec.pool.items():
            if key == src_set:
                continue

            for user, sess in d.train:
                if f"{user}-{sess}" not in exclude_list:
                    seq_list.append([user, sess])
            for user, sess in d.test:
                if f"{user}-{sess}" not in exclude_list:
                    seq_list.append([user, sess])
    else:
        raise NotImplementedError(f"stage={stage} is not supported.")

    return seq_list


def assemble_sequence_list_flexible_train_data_volume_setting(
    cfg: DictConfig, stage: str
):
    assert cfg.dataset.split.kind == "dataset/split/flexible-train-data-volume"

    train_set = cfg.metadata.labels.train_set

    if stage in ("fit", "train"):
        seq_list = []

        pool_keys = sorted(list(cfg.dataset.split.spec.pool.keys()))
        log.debug(f"pool_keys={pool_keys}")
        ind = pool_keys.index(train_set)
        for key in pool_keys[: (ind + 1)]:
            seq_list += cfg.dataset.split.spec.pool[key].train
    elif stage == "test":
        seq_list = cfg.dataset.split.spec.test
    else:
        raise NotImplementedError(f"stage={stage} is not supported.")

    return seq_list


def assemble_sequence_list_leave_one_out_setting(cfg: DictConfig, stage: str):
    assert cfg.dataset.split.kind == "dataset/split/leave-one-out-cv"

    test_set = cfg.metadata.labels.test_set

    if stage in ("fit", "train"):
        seq_list = []

        pool_keys = sorted(list(cfg.dataset.split.spec.pool.keys()))
        log.debug(f"pool_keys={pool_keys}")

        seq_list = []
        for key, _seq_list in cfg.dataset.split.spec.pool.items():
            if key != test_set:
                seq_list += list(_seq_list)
    elif stage == "test":
        seq_list = cfg.dataset.split.spec.pool[test_set]
    else:
        raise NotImplementedError(f"stage={stage} is not supported.")

    return seq_list


def assemble_sequence_list_from_cfg(cfg: DictConfig, stage: str):
    split_kind = cfg.dataset.split.kind
    if split_kind == "dataset/split/data-volume-flexible-cv":
        seq_list = assemble_sequence_list_data_volume_flexible_cv(cfg, stage)
    elif split_kind == "dataset/split/flexible-train-data-volume":
        seq_list = assemble_sequence_list_flexible_train_data_volume_setting(cfg, stage)
    elif split_kind == "dataset/split/leave-one-out-cv":
        seq_list = assemble_sequence_list_leave_one_out_setting(cfg, stage)
    else:
        raise ValueError(f"unknown split type: {cfg.dataset.split.kind}")

    return seq_list


# -----------------------------------------------------------------------------


def split_dataset(
    cfg: DictConfig,
    dataset: torch.utils.data.Dataset,
    val_split_size: float = 0.2,
):
    """Split this instance into train and val dataset class.

    Note:
        Some leakage occurs between train and test with ``random_crop=True``.
    """
    assert val_split_size > 0.0

    # Split Index
    original_index = dict()
    for win in dataset.index:
        if win.sequence_idx in original_index.keys():
            original_index[win.sequence_idx] += [win]
        else:
            original_index[win.sequence_idx] = [win]

    train_index, val_index = [], []
    keys = sorted(original_index.keys())
    for key in keys:
        num_sample = len(original_index[key])
        num_train = int(num_sample * (1.0 - val_split_size))
        train_index += original_index[key][:num_train]
        val_index += original_index[key][num_train:]

    # Split
    dataset_train = dataset.__class__(
        cfg,
        user_session_list=None,
        classes=dataset.classes,
        window=dataset.window,
        random_crop=dataset.random_crop,
        submission=dataset.submission,
        debug=dataset.debug,
    )
    dataset_train.data = dataset.data.copy()
    dataset_train.index = train_index

    dataset_val = dataset.__class__(
        cfg,
        user_session_list=None,
        classes=dataset.classes,
        window=dataset.window,
        random_crop=False,
        submission=dataset.submission,
        debug=dataset.debug,
    )
    dataset_val.data = dataset.data.copy()
    dataset_val.index = val_index

    return dataset_train, dataset_val

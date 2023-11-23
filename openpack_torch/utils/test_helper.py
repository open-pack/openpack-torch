from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from openpack_toolkit import ActSet
from openpack_toolkit.codalab.operation_segmentation import (
    eval_operation_segmentation_wrapper,
)
from openpack_toolkit.configs.datasets.annotations import (
    OPENPACK_ACTIONS,
    OPENPACK_OPERATIONS,
)

from openpack_torch.data.utils import assemble_sequence_list_from_cfg

logger = getLogger(__name__)

SCENARIO_DICT = {
    "U0101-S0100": "S1",
    "U0101-S0200": "S1",
    "U0101-S0300": "S1",
    "U0101-S0400": "S1",
    "U0101-S0500": "S1",
    "U0102-S0100": "S1",
    "U0102-S0200": "S1",
    "U0102-S0300": "S1",
    "U0102-S0400": "S1",
    "U0102-S0500": "S1",
    "U0103-S0100": "S1",
    "U0103-S0200": "S1",
    "U0103-S0300": "S1",
    "U0103-S0400": "S1",
    "U0103-S0500": "S1",
    "U0104-S0100": "S1",
    "U0104-S0200": "S1",
    "U0104-S0300": "S1",
    "U0104-S0400": "S1",
    "U0105-S0100": "S1",
    "U0105-S0200": "S1",
    "U0105-S0300": "S1",
    "U0105-S0400": "S1",
    "U0105-S0500": "S1",
    "U0106-S0100": "S1",
    "U0106-S0200": "S1",
    "U0106-S0300": "S1",
    "U0106-S0400": "S1",
    "U0106-S0500": "S1",
    "U0107-S0100": "S1",
    "U0107-S0200": "S1",
    "U0107-S0300": "S1",
    "U0107-S0400": "S1",
    "U0107-S0500": "S1",
    "U0108-S0100": "S1",
    "U0108-S0200": "S1",
    "U0108-S0300": "S1",
    "U0108-S0400": "S1",
    "U0108-S0500": "S1",
    "U0109-S0100": "S1",
    "U0109-S0200": "S1",
    "U0109-S0300": "S1",
    "U0109-S0400": "S1",
    "U0109-S0500": "S1",
    "U0110-S0100": "S1",
    "U0110-S0200": "S1",
    "U0110-S0300": "S1",
    "U0110-S0400": "S1",
    "U0110-S0500": "S1",
    "U0111-S0100": "S1",
    "U0111-S0200": "S1",
    "U0111-S0300": "S1",
    "U0111-S0400": "S1",
    "U0111-S0500": "S1",
    "U0201-S0100": "S2",
    "U0201-S0200": "S2",
    "U0201-S0300": "S3",
    "U0201-S0400": "S3",
    "U0201-S0500": "S4",
    "U0202-S0100": "S2",
    "U0202-S0200": "S2",
    "U0202-S0300": "S3",
    "U0202-S0400": "S3",
    "U0202-S0500": "S4",
    "U0203-S0100": "S2",
    "U0203-S0200": "S2",
    "U0203-S0300": "S3",
    "U0203-S0400": "S3",
    "U0203-S0500": "S4",
    "U0204-S0100": "S2",
    "U0204-S0200": "S2",
    "U0204-S0300": "S3",
    "U0204-S0400": "S3",
    "U0204-S0500": "S4",
    "U0205-S0100": "S2",
    "U0205-S0200": "S2",
    "U0205-S0300": "S3",
    "U0205-S0400": "S3",
    "U0205-S0500": "S4",
    "U0206-S0100": "S2",
    "U0206-S0200": "S2",
    "U0206-S0300": "S3",
    "U0206-S0400": "S3",
    "U0206-S0500": "S4",
    "U0207-S0100": "S2",
    "U0207-S0200": "S2",
    "U0207-S0300": "S3",
    "U0207-S0400": "S3",
    "U0207-S0500": "S4",
    "U0208-S0100": "S2",
    "U0208-S0200": "S2",
    "U0208-S0300": "S3",
    "U0208-S0400": "S3",
    "U0208-S0500": "S4",
    "U0209-S0100": "S2",
    "U0209-S0200": "S2",
    "U0209-S0300": "S3",
    "U0209-S0400": "S3",
    "U0209-S0500": "S4",
    "U0210-S0100": "S2",
    "U0210-S0200": "S2",
    "U0210-S0300": "S3",
    "U0210-S0400": "S3",
    "U0210-S0500": "S4",
}


def test_helper(
    cfg: DictConfig,
    mode: str,
    datamodule: pl.LightningDataModule,
    plmodel: pl.LightningModule,
    trainer: pl.Trainer,
):
    if cfg.dataset.annotation.name == "openpack-actions-1hz-annotation":
        classes = ActSet(OPENPACK_ACTIONS)
    elif cfg.dataset.annotation.name == "openpack-operations-1hz-annotation":
        classes = ActSet(OPENPACK_OPERATIONS)
    else:
        raise ValueError(f"{cfg.dataset.annotation.name} is not supported.")

    dataloaders = None
    if cfg.metadata.labels.benchmarkType == "benchmark1":
        if mode == "test":
            dataloaders = datamodule.test_dataloader()
            if hasattr(cfg.dataset.split, "spec"):
                split = cfg.dataset.split.spec.test
            else:
                split = cfg.dataset.split.test
        elif mode in ("submission", "test-on-submission"):
            dataloaders = datamodule.submission_dataloader()
            if hasattr(cfg.dataset.split, "spec"):
                split = cfg.dataset.split.spec.submission
            else:
                split = cfg.dataset.split.submission

    elif cfg.metadata.labels.benchmarkType in (
        "benchmark2",
        "benchmark3",
        "benchmark5",
    ):
        if mode == "test":
            dataloaders = datamodule.test_dataloader()
            split = [s.split("-") for s in datamodule.dataset_test.keys()]
    if dataloaders is None:
        raise NotImplementedError(
            f"mode={mode} on benchmark={cfg.metadata.labels.benchmarkType} is not supported."
        )

    outputs = dict()
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            fname = key.replace("/", "-")
            path = Path(pred_dir, f"{fname}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        key = f"{user}-{session}"
        outputs[key] = {
            "y": plmodel.test_results.get("y"),
            "unixtime": plmodel.test_results.get("unixtime"),
        }
        if mode in ("test", "test-on-submission"):
            outputs[key].update(
                {
                    "t_idx": plmodel.test_results.get("t"),
                }
            )

    df_summary = None
    if cfg.metadata.labels.benchmarkType == "benchmark1":
        if mode in ("test", "test-on-submission"):
            split = [k.split("-") for k in outputs.keys()]
            df_summary = compute_score_for_each_scenario(cfg, classes, split, outputs)

            if mode == "test":
                path = Path(cfg.path.logdir.summary.test)
            elif mode == "test-on-submission":
                path = Path(cfg.path.logdir.summary.submission)
            path.parent.mkdir(parents=True, exist_ok=True)
            df_summary.to_csv(path, index=False)
            logger.info(f"write df_summary[shape={df_summary.shape}] to {path}")

            # NOTE: change pandas option to show tha all rows/cols.
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            logger.info(f"df_summary:\n{df_summary}")
        elif mode == "submission":
            raise NotImplementedError()
    elif cfg.metadata.labels.benchmarkType == "benchmark2":
        assert mode == "test"

        for stage in ["train", "test-b2"]:
            logger.info(f"test on stage={stage}")

            split = assemble_sequence_list_from_cfg(cfg, stage)
            df_summary = compute_score_for_each_scenario(cfg, classes, split, outputs)
            path = Path(cfg.path.logdir.summary[stage])
            path.parent.mkdir(parents=True, exist_ok=True)
            df_summary.to_csv(path, index=False)
            logger.info(f"write df_summary[shape={df_summary.shape}] to {path}")

            # NOTE: change pandas option to show tha all rows/cols.
            # pd.set_option('display.max_rows', None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            logger.info(f"df_summary:\n{df_summary.tail(15)}")

    elif cfg.metadata.labels.benchmarkType == "benchmark3":
        # FIXME: per scenario score computation
        assert mode == "test"

        # TODO: Prepare stage=train
        for stage in ["test"]:
            logger.info(f"test on stage={stage}")

            split = assemble_sequence_list_from_cfg(cfg, stage)
            df_summary = compute_score_for_each_scenario(cfg, classes, split, outputs)

            path = Path(cfg.path.logdir.summary[stage])
            path.parent.mkdir(parents=True, exist_ok=True)
            df_summary.to_csv(path, index=False)
            logger.info(f"write df_summary[shape={df_summary.shape}] to {path}")

            # NOTE: change pandas option to show tha all rows/cols.
            # pd.set_option('display.max_rows', None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            logger.info(f"df_summary:\n{df_summary.tail(15)}")

    elif cfg.metadata.labels.benchmarkType == "benchmark5":
        assert mode == "test"

        for stage in ["test"]:
            logger.info(f"test on stage={stage}")

            split = assemble_sequence_list_from_cfg(cfg, stage)
            df_summary = compute_score_for_each_scenario(cfg, classes, split, outputs)
            path = Path(cfg.path.logdir.summary[stage])
            path.parent.mkdir(parents=True, exist_ok=True)
            df_summary.to_csv(path, index=False)
            logger.info(f"write df_summary[shape={df_summary.shape}] to {path}")

            # NOTE: change pandas option to show tha all rows/cols.
            # pd.set_option('display.max_rows', None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            logger.info(f"df_summary:\n{df_summary.tail(15)}")

    return outputs, df_summary


def compute_score_for_each_scenario(cfg, classes, split, outputs):
    df_summary = []

    for target_scenario in ["S1", "S2", "S3", "S4", "all"]:
        _outputs = dict()
        for _user, _sess in split:
            key = f"{_user}-{_sess}"
            scenario = SCENARIO_DICT[key]
            if (scenario == target_scenario) or (target_scenario == "all"):
                _outputs[key] = outputs[key]

        if len(_outputs) > 0:
            df_tmp = eval_operation_segmentation_wrapper(cfg, _outputs, classes)
            df_tmp["scenario"] = target_scenario
            df_summary.append(df_tmp)

    df_summary = pd.concat(df_summary, axis=0)
    return df_summary

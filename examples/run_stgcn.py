from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import openpack_toolkit as optk
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict,
    eval_operation_segmentation_wrapper,
    make_submission_zipfile,
)

import openpack_torch as optorch
from openpack_torch.lightning import EarlyStopError
from openpack_torch.utils.test_helper import test_helper

logger = getLogger(__name__)
optorch.configs.register_configs()
optorch.utils.reset_seed(seed=0)


# ----------------------------------------------------------------------
class OpenPackKeypointDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.Kinect2dKptDataset

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "debug": self.cfg.debug,
            "window": self.cfg.train.window,
        }
        return kwargs


class STGCN4SegLM(optorch.lightning.BaseLightningModule):
    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        input_dim = 3
        output_dim = len(self.cfg.dataset.annotation.spec.classes)

        Ks = cfg.model.spec.Ks
        A = optorch.models.keypoint.get_adjacency_matrix(
            layout="MSCOCO", hop_size=Ks - 1
        )
        model = optorch.models.keypoint.STGCN4Seg(
            input_dim,
            output_dim,
            Ks=cfg.model.spec.Ks,
            Kt=cfg.model.spec.Kt,
            A=A,
        )
        return model

    def train_val_common_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)

        y_hat = self(x).squeeze(3)

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat = self(x).squeeze(3)

        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        self.test_step_outputs.append(outputs)
        return outputs


# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    optk.utils.io.cleanup_dir(logdir, exclude="hydra")

    datamodule = OpenPackKeypointDataModule(cfg)
    plmodel = STGCN4SegLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    logger.info(plmodel)

    max_epoch = (
        cfg.train.debug.epochs.maximum if cfg.debug else cfg.train.epochs.maximum
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        mode=cfg.train.early_stop.mode,
        monitor=cfg.train.early_stop.monitor,
        filename="{epoch:02d}-{train/loss:.2f}-{val/loss:.2f}",
        verbose=False,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        **cfg.train.early_stop,
    )

    pl_logger = pl.loggers.CSVLogger(logdir)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=max_epoch,
        logger=pl_logger,
        default_root_dir=logdir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=4,
    )

    logger.info(f"Start training for {max_epoch} epochs.")
    try:
        trainer.fit(plmodel, datamodule)
    except EarlyStopError as e:
        logger.warning(e)
    logger.info(f"Finish training! (logdir = {logdir})")


def test(cfg: DictConfig, mode: str = "test"):
    assert mode in ("test", "submission", "test-on-submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)

    datamodule = OpenPackKeypointDataModule(cfg)
    datamodule.setup(mode)

    if cfg.train.checkpoint == "best":
        raise NotImplementedError()
    elif cfg.train.checkpoint == "last":
        ckpt_path = Path(
            logdir, "lightning_logs", "version_0", "checkpoints", "last.ckpt"
        )
    else:
        raise ValueError()
    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = STGCN4SegLM.load_from_checkpoint(ckpt_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    test_helper(cfg, mode, datamodule, plmodel, trainer)


@hydra.main(version_base=None, config_path="./configs", config_name="st-gcn.yaml")
def main(cfg: DictConfig):
    # DEBUG
    if cfg.debug:
        cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
        cfg.path.logdir.rootdir += "/debug"

    print("===== Params =====")
    print(OmegaConf.to_yaml(cfg))
    print("==================")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test", "submission", "test-on-submission"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")


if __name__ == "__main__":
    main()

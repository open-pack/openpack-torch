from logging import getLogger
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict, eval_operation_segmentation_wrapper,
    make_submission_zipfile)

logger = getLogger(__name__)

# ----------------------------------------------------------------------


def save_training_results(log: Dict, logdir: Path) -> None:
    # -- Save Model Outputs --
    df = pd.concat(
        [
            pd.DataFrame(log["train"]),
            pd.DataFrame(log["val"]),
        ],
        axis=1,
    )
    df.index.name = "epoch"

    path = Path(logdir, "training_log.csv")
    df.to_csv(path, index=True)
    logger.debug(f"Save training logs to {path}")
    print(df)


# ----------------------------------------------------------------------
class OpenPackKeypointDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.OpenPackKeypoint

    def get_kwargs_for_datasets(self) -> Dict:
        kpt_cfg = self.cfg.dataset.modality.keypoint
        submission = True if self.cfg.mode == "submission" else False

        kwargs = {
            "keypoint_type": kpt_cfg.type,
            "debug": self.cfg.debug,
            "window": self.cfg.train.window,
            "submission": submission,
        }
        return kwargs


class STGCN4SegLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        if cfg.dataset.modality.keypoint.format == "2D":
            in_ch = 2
        elif cfg.dataset.modality.keypoint.format == "3D":
            in_ch = 3
        else:
            raise ValueError()

        Ks = cfg.model.Ks
        A = optorch.models.keypoint.get_adjacency_matrix(
            layout="MSCOCO", hop_size=Ks - 1)
        model = optorch.models.keypoint.STGCN4Seg(
            in_ch,
            cfg.dataset.num_classes,
            Ks=cfg.model.Ks,
            Kt=cfg.model.Kt,
            A=A,
        )
        return model

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
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
        return outputs


# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path.cwd()
    logger.debug(f"logdir = {logdir}")
    optk.utils.io.cleanup_dir(logdir, exclude="hydra")

    datamodule = OpenPackKeypointDataModule(cfg)
    plmodel = STGCN4SegLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    logger.info(plmodel)

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor=None,
    )

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=num_epoch,
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    logger.debug(f"logdir = {logdir}")

    logger.info(f"Train the model for {num_epoch} epochs.")
    trainer.fit(plmodel, datamodule)
    logger.info("Finish training!")

    logger.debug(f"logdir = {logdir}")
    save_training_results(plmodel.log, logdir)
    logger.debug(f"logdir = {logdir}")


def test(cfg: DictConfig, mode: str = "test"):
    assert mode in ("test", "submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda")
    logdir = Path(cfg.volume.logdir.rootdir)

    datamodule = OpenPackKeypointDataModule(cfg)
    datamodule.setup(mode)

    ckpt_path = Path(logdir, "checkpoints", "last.ckpt")
    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = STGCN4SegLM.load_from_checkpoint(ckpt_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)

    trainer = pl.Trainer(
        gpus=[0],
        logger=False,  # disable logging module
        default_root_dir=None,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    if mode == "test":
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test
    elif mode == "submission":
        dataloaders = datamodule.submission_dataloader()
        split = cfg.dataset.split.submission
    outputs = dict()
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on U{user:0=4}-S{session:0=4}")

        trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(
            cfg.volume.logdir.predict.format(user=user, session=session)
        )
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            path = Path(pred_dir, f"{key}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        key = f"U{user:0=4}-S{session:0=4}"
        outputs[key] = {
            "y": plmodel.test_results.get("y"),
            "unixtime": plmodel.test_results.get("unixtime"),
        }
        if mode == "test":
            outputs[key].update({
                "t_idx": plmodel.test_results.get("t"),
            })

    if mode == "test":
        # save performance summary
        df_summary = eval_operation_segmentation_wrapper(
            outputs, OPENPACK_OPERATIONS,
        )
        path = Path(cfg.volume.logdir.summary)
        df_summary.to_csv(path, index=False)
        logger.info(f"df_summary:\n{df_summary}")
    elif mode == "submission":
        # make submission file
        submission_dict = construct_submission_dict(
            outputs, OPENPACK_OPERATIONS)
        make_submission_zipfile(submission_dict, logdir)


@ hydra.main(version_base=None, config_path="../../configs",
             config_name="operation-segmentation-stgcn.yaml")
def main(cfg: DictConfig):
    print("===== Params =====")
    print(OmegaConf.to_yaml(cfg))
    print("==================")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test", "submission"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")


if __name__ == "__main__":
    main()

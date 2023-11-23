from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.functional import accuracy as accuracy_score

logger = getLogger(__name__)


class EarlyStopError(Exception):
    pass


class BaseLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig = None) -> None:
        self.cfg = cfg
        super().__init__()

        self.net: nn.Module = self.init_model(cfg)
        self.criterion: nn.Module = self.init_criterion(cfg)

        self.test_step_outputs: List = []

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        raise NotImplementedError()

    def init_criterion(self, cfg: DictConfig):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # == Optimizer ==
        if self.cfg.optimizer.type == "SGD":
            logger.info(f"SGD optimizer is selected! (lr={self.cfg.optimizer.lr})")
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif self.cfg.optimizer.type == "Adam":
            logger.info(f"Adam optimizer is selected! (lr={self.cfg.optimizer.lr})")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"{self.cfg.optimizer.type} is not supported.")

        # == LR Scheduler ==
        if self.cfg.optimizer.scheduler.type == "None":
            logger.info("No scheduler is applied.")
            return optimizer
        elif self.cfg.optimizer.scheduler.type == "StepLR":
            logger.info("StepLR scheduler is selected.")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.optimizer.scheduler.step_size,
                gamma=self.cfg.optimizer.scheduler.gamma,
            )
        elif self.cfg.optimizer.scheduler.type == "ExponentialLR":
            logger.info("StepLR scheduler is selected.")
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.cfg.optimizer.scheduler.gamma,
            )
        elif self.cfg.optimizer.scheduler.type == "CosineAnnealing":
            logger.info("CosineAnnealing scheduler is selected.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optimizer.scheduler.CosineAnnealing.T_max,
                eta_min=self.cfg.optimizer.scheduler.CosineAnnealing.eta_min,
                verbose=True,
            )
        else:
            raise ValueError()

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def calc_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns accuracy score.

        Args:
            y (torch.Tensor): logit tensor. shape=(BATCH, CLASS, TIME), dtype=torch.float
            t (torch.Tensor): target tensor. shape=(BATCH, TIME), dtype=torch.long

        Returns:
            torch.Tensor: _description_
        """
        preds = F.softmax(y, dim=1)
        (batch_size, num_classes, window_size) = preds.size()
        preds_flat = preds.permute(1, 0, 2).reshape(
            num_classes, batch_size * window_size
        )
        t_flat = t.reshape(-1)

        # FIXME: I want to use macro average score.
        ignore_index = num_classes - 1
        acc = accuracy_score(
            preds_flat.transpose(0, 1),
            t_flat,
            task="multiclass",
            average="weighted",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_val_common_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        output = self.train_val_common_step(batch, batch_idx)

        train_output = {f"train/{key}": val for key, val in output.items()}
        self.log_dict(
            train_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return output

    def validation_step(
        self, batch: Dict, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict:
        output = self.train_val_common_step(batch, batch_idx)

        train_output = {f"val/{key}": val for key, val in output.items()}
        self.log_dict(
            train_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return output

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            raise ValueError(
                "Size of test_step_outputs is 0. Did you forgot to call "
                "`self.test_step_outputs.append(outputs)` in test_step()?"
            )
        outputs = self.test_step_outputs

        keys = tuple(outputs[0].keys())
        results = {key: [] for key in keys}
        for d in outputs:
            for key in d.keys():
                results[key].append(d[key].cpu().numpy())

        for key in keys:
            results[key] = np.concatenate(results[key], axis=0)

        self.test_results = results

    def clear_test_outputs(self):
        self.test_step_outputs = []
        self.test_results = None

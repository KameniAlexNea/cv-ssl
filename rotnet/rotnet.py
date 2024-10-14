import argparse
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import torch
import torch.nn as nn

from rotnet.loss import rotnet_loss_fun
from rotnet.base import BaseModel


class RotNet(BaseModel):
    def __init__(
        self,
        num_rotation: int,
        output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        rot_loss_weight: float,
        **kwargs
    ):
        """Implements RotNet

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            rot_loss_weight (float): weight of the variance term.
        """

        super().__init__(**kwargs)

        self.num_rotation = num_rotation
        self.output_dim = output_dim

        self.sim_loss_weight = sim_loss_weight
        self.rot_loss_weight = rot_loss_weight

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, num_rotation),
        )

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parent_parser = super(RotNet, RotNet).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("vicreg")

        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)


        # parameters
        parser.add_argument("--sim_loss_weight", default=1, type=float)
        parser.add_argument("--rot_loss_weight", default=1, type=float)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"head": self.projector.parameters()}, {"sim_projector": self.sim_projection}]
        return super().learnable_params + extra_learnable_params

    def training_step(self, batch: Sequence[Any], batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
        """Training step for RotNet reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of RotNet loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        if optimizer_idx:
            return out
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- rotnet loss function -------
        sim_loss, rotnet_loss = rotnet_loss_fun(
            z1,
            z2,
            feats1,
            feats2
        )

        loss = rotnet_loss * self.rot_loss_weight + sim_loss * self.sim_loss_weight

        self.log("train_rotnet_loss", rotnet_loss, on_epoch=True, sync_dist=True)
        self.log("train_sim_loss", sim_loss, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + loss, "z": [z1, z2]})
        return out

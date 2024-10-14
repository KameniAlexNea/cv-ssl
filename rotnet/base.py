import functools
import operator
from argparse import ArgumentParser
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR

from rotnet.lars import LARSWrapper
from rotnet.metrics import accuracy_at_k
from rotnet.metrics import weighted_mean


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder: str,
        num_classes: int,
        cifar: bool,
        zero_init_residual: bool,
        max_epochs: int,
        batch_size: int,
        online_eval_batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: int,
        extra_optimizer_args: Dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        multicrop: bool,
        num_crops: int,
        num_small_crops: int,
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            encoder (str): architecture of the base encoder.
            num_classes (int): number of classes.
            cifar (bool): flag indicating if cifar is being used.
            zero_init_residual (bool): change the initialization of the resnet encoder.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (int): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            multicrop (bool): flag indicating if multi-resolution crop is being used.
            num_crops (int): number of big crops
            num_small_crops (int): number of small crops (will be set to 0 if multicrop is False).
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
        """

        super().__init__()

        # back-bone related
        self.cifar = cifar
        self.zero_init_residual = zero_init_residual

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.online_eval_batch_size = online_eval_batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.multicrop = multicrop
        self.num_crops = num_crops
        self.num_small_crops = num_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars

        # sanity checks on multicrop
        if self.multicrop:
            assert num_small_crops > 0
        else:
            self.num_small_crops = 0

        # check if should perform online eval
        self.online_eval = online_eval_batch_size is not None

        # all the other parameters
        self.extra_args = kwargs

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert encoder in ["resnet18", "resnet50"]
        from torchvision.models import resnet18
        from torchvision.models import resnet50

        self.base_model = {"resnet18": resnet18, "resnet50": resnet50}[encoder]

        # initialize encoder
        self.encoder = self.base_model(zero_init_residual=zero_init_residual)
        self.features_dim = self.encoder.inplanes
        # remove fc layer
        self.encoder.fc = nn.Identity()
        if cifar:
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.encoder.maxpool = nn.Identity()

        self.classifier = nn.Linear(self.features_dim, num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # encoder args
        SUPPORTED_NETWORKS = ["resnet18", "resnet50"]

        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
        parser.add_argument("--zero_init_residual", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

        parser.add_argument(
            "--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True
        )
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument(
            "--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps",
                            default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # DALI only
        # uses sample indexes as labels and then gets the labels from a lookup table
        # this may use more CPU memory, so just use when needed.
        parser.add_argument("--encode_indexes_into_labels",
                            action="store_true")

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        lr_optimizer = torch.optim.Adam(
            [i for i in self.learnable_params if i.get("name", None) == "classifier"],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # optionally wrap with lars
        if self.lars:
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )
            lr_optimizer = LARSWrapper(
                lr_optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return [optimizer, lr_optimizer]
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer, self.max_epochs, eta_min=self.min_lr
                )
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(
                    f"{self.scheduler} not in (warmup_cosine, cosine, step)"
                )

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer, lr_optimizer], [scheduler]

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""

        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> torch.Tensor:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            torch.Tensor: features extracted by the encoder.
        """

        return {"feats": self.encoder(X)}

    def _online_eval_shared_step(self, X: torch.Tensor, targets) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format
            targets (torch.Tensor): batch of labels for X

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5
        """

        with torch.no_grad():
            outs = self.base_forward(X)
        feats = outs["feats"].detach()
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))

        return {
            **outs,
            "logits": logits,
            "loss": loss,
            "acc1": acc1.detach(),
            "acc5": acc5.detach(),
        }

    def training_step(self, batch: List[Any], batch_idx: int, optimizer_idx: int = 0) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits
        """

        if optimizer_idx == 0:
            _, X_task, _ = batch
            X_task = [X_task] if isinstance(X_task, torch.Tensor) else X_task

            # check that we received the desired number of crops
            assert len(X_task) == self.num_crops + self.num_small_crops

            # forward views of the current task in the encoder
            outs_task = [self.base_forward(x) for x in X_task[: self.num_crops]]
            outs_task = {k: [out[k] for out in outs_task]
                        for k in outs_task[0].keys()}

            if self.multicrop:
                outs_task["feats"].extend(
                    [self.encoder(x) for x in X_task[self.num_crops:]]
                )

        if self.online_eval and optimizer_idx == 1:
            assert "online_eval" in batch.keys()
            *_, X_online_eval, targets_online_eval = batch["online_eval"]

            # forward online eval images and calculate online eval loss
            outs_online_eval = self._online_eval_shared_step(
                X_online_eval, targets_online_eval
            )
            outs_online_eval = {
                "online_eval_" + k: v for k, v in outs_online_eval.items()
            }

            metrics = {
                "train_online_eval_loss": outs_online_eval["online_eval_loss"],
                "train_online_eval_acc1": outs_online_eval["online_eval_acc1"],
                "train_online_eval_acc5": outs_online_eval["online_eval_acc5"],
            }

            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            loss = outs_online_eval.pop("online_eval_loss")
            return {**outs_online_eval, **{"loss": loss}}
        else:
            return {**outs_task, "loss": 0}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y]
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies
        """

        if self.online_eval:
            *_, X, targets = batch

            batch_size = targets.size(0)

            out = self._online_eval_shared_step(X, targets)

            metrics = {
                "batch_size": batch_size,
                "targets": targets,
                "val_loss": out["loss"],
                "val_acc1": out["acc1"],
                "val_acc5": out["acc5"],
            }

            return {**metrics, **out}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        if self.online_eval:
            val_loss = weighted_mean(outs, "val_loss", "batch_size")
            val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
            val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

            log = {"val_loss": val_loss,
                   "val_acc1": val_acc1, "val_acc5": val_acc5}

            self.log_dict(log, sync_dist=True)

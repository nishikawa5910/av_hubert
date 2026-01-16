# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class ContentVecMSECriterionConfig(FairseqDataclass):
    mse_loss_weight: float = field(
        default=1.0, metadata={"help": "weight for contentvec MSE loss"}
    )


@register_criterion("contentvec_mse", dataclass=ContentVecMSECriterionConfig)
class ContentVecMSECriterion(FairseqCriterion):
    def __init__(self, task, mse_loss_weight=1.0):
        super().__init__(task)
        self.mse_loss_weight = mse_loss_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        pred = net_output["pred"]  # T x B x C
        target = sample["target"]  # B x T x C
        target = target.transpose(0, 1)

        if "target_lengths" in sample:
            lengths = sample["target_lengths"]
            max_len = target.size(0)
            mask = torch.arange(max_len, device=lengths.device).unsqueeze(1) < lengths.unsqueeze(0)
            mask = mask.unsqueeze(-1)
            pred = pred * mask
            target = target * mask
            denom = (mask.sum() * pred.size(-1)).clamp_min(1).float()
        else:
            denom = torch.tensor(pred.numel(), device=pred.device, dtype=torch.float)

        loss = F.mse_loss(pred, target, reduction="sum") / denom
        loss = loss * self.mse_loss_weight

        sample_size = target.size(1)
        logging_output: Dict[str, float] = {
            "loss": loss.detach().item(),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


@dataclass
class ContentVecCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0, metadata={"help": "label smoothing for class targets"}
    )


@register_criterion(
    "contentvec_ce", dataclass=ContentVecCrossEntropyCriterionConfig
)
class ContentVecCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing=0.0):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        pred = net_output["pred"]  # T x B x C
        if "target" in sample:
            target = sample["target"]  # B x T
        else:
            target = sample["target_list"][0]  # B x T
        target = target.transpose(0, 1)

        pred = pred.transpose(0, 1).contiguous().view(-1, pred.size(-1))
        target = target.contiguous().view(-1)

        padding_mask: Optional[torch.Tensor] = net_output.get("padding_mask")
        if padding_mask is not None:
            mask = ~padding_mask.transpose(0, 1).reshape(-1)
            if mask.numel() == 0 or target.numel() == 0:
                loss = pred.sum() * 0.0
                sample_size = 0
                logging_output = {"loss": 0.0, "sample_size": 0}
                return loss, sample_size, logging_output
            pred = pred[mask]
            target = target[mask]
            if target.numel() == 0:
                loss = pred.sum() * 0.0
                sample_size = 0
                logging_output = {"loss": 0.0, "sample_size": 0}
                return loss, sample_size, logging_output

        if self.task.target_dictionary is not None:
            ignore_index = self.task.target_dictionary.pad()
        else:
            ignore_index = -1

        loss = F.cross_entropy(
            pred,
            target.long(),
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )
        sample_size = target.numel()

        logging_output: Dict[str, float] = {
            "loss": loss.detach().item(),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

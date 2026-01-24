# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)


@dataclass
class LabelSmoothedCrossEntropyWithContentVecConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    contentvec_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for auxiliary contentvec loss"},
    )
    contentvec_label_idx: int = field(
        default=1,
        metadata={"help": "index of contentvec labels in task.labels"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_contentvec",
    dataclass=LabelSmoothedCrossEntropyWithContentVecConfig,
)
class LabelSmoothedCrossEntropyWithContentVecCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        contentvec_loss_weight=1.0,
        contentvec_label_idx=1,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.contentvec_loss_weight = contentvec_loss_weight
        self.contentvec_label_idx = contentvec_label_idx

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output: Dict[str, float] = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample.get("ntokens", 0),
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        contentvec_pred, contentvec_padding_mask = self._get_contentvec_pred(net_output)
        contentvec_target, contentvec_lengths = self._get_contentvec_targets(sample)

        if contentvec_pred is not None and contentvec_target is not None:
            contentvec_loss, contentvec_sample_size = self._compute_contentvec_loss(
                model,
                contentvec_pred,
                contentvec_target,
                contentvec_lengths,
                contentvec_padding_mask,
            )
            loss = loss + self.contentvec_loss_weight * contentvec_loss
            logging_output["contentvec_loss"] = contentvec_loss.data
            logging_output["contentvec_sample_size"] = contentvec_sample_size

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = n_correct
            logging_output["total"] = total

        return loss, sample_size, logging_output

    def _get_contentvec_pred(self, net_output) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(net_output, tuple) and len(net_output) > 1 and isinstance(net_output[1], dict):
            return (
                net_output[1].get("contentvec_pred"),
                net_output[1].get("contentvec_padding_mask"),
            )
        return None, None

    def _get_contentvec_targets(
        self, sample
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if "audio_targets" in sample:
            return sample["audio_targets"], sample.get("audio_target_lengths")
        target_list = sample.get("target_list")
        if target_list is None:
            return None, None
        if len(target_list) <= self.contentvec_label_idx:
            return None, None
        target = target_list[self.contentvec_label_idx]
        lengths = None
        if "target_lengths_list" in sample and len(sample["target_lengths_list"]) > self.contentvec_label_idx:
            lengths = sample["target_lengths_list"][self.contentvec_label_idx]
        return target, lengths

    def _contentvec_target_type(self, model) -> str:
        return getattr(getattr(model, "cfg", None), "contentvec_target_type", "class")

    def _get_contentvec_pad_idx(self) -> int:
        dictionaries = getattr(self.task, "dictionaries", None)
        if dictionaries is not None and len(dictionaries) > self.contentvec_label_idx:
            dictionary = dictionaries[self.contentvec_label_idx]
            if dictionary is not None:
                return dictionary.pad()
        return self.padding_idx

    def _compute_contentvec_loss(
        self,
        model,
        contentvec_pred: torch.Tensor,
        contentvec_target: torch.Tensor,
        contentvec_lengths: Optional[torch.Tensor],
        contentvec_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        target_type = self._contentvec_target_type(model)
        if target_type == "float":
            return self._compute_contentvec_mse(
                contentvec_pred, contentvec_target, contentvec_lengths
            )
        return self._compute_contentvec_ce(
            contentvec_pred,
            contentvec_target,
            contentvec_lengths,
            contentvec_padding_mask,
        )

    def _compute_contentvec_mse(
        self,
        contentvec_pred: torch.Tensor,
        contentvec_target: torch.Tensor,
        contentvec_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        target = contentvec_target.transpose(0, 1)
        pred = contentvec_pred
        if contentvec_lengths is not None:
            max_len = target.size(0)
            mask = (
                torch.arange(max_len, device=contentvec_lengths.device)
                .unsqueeze(1)
                .expand(max_len, contentvec_lengths.size(0))
                < contentvec_lengths.unsqueeze(0)
            )
            mask = mask.unsqueeze(-1)
            pred = pred * mask
            target = target * mask
            denom = (mask.sum() * pred.size(-1)).clamp_min(1).float()
        else:
            denom = torch.tensor(pred.numel(), device=pred.device, dtype=torch.float)

        loss = F.mse_loss(pred, target, reduction="sum") / denom
        sample_size = target.size(1)
        return loss, sample_size

    def _compute_contentvec_ce(
        self,
        contentvec_pred: torch.Tensor,
        contentvec_target: torch.Tensor,
        contentvec_lengths: Optional[torch.Tensor],
        contentvec_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        pred = contentvec_pred.transpose(0, 1)
        target = contentvec_target
        pred = pred.contiguous().view(-1, pred.size(-1))
        target = target.contiguous().view(-1)

        mask = None
        if contentvec_lengths is not None:
            max_len = contentvec_pred.size(0)
            length_mask = (
                torch.arange(max_len, device=contentvec_lengths.device)
                .unsqueeze(1)
                .expand(max_len, contentvec_lengths.size(0))
                < contentvec_lengths.unsqueeze(0)
            )
            mask = length_mask.transpose(0, 1).reshape(-1)
        elif contentvec_padding_mask is not None:
            mask = ~contentvec_padding_mask.transpose(0, 1).reshape(-1)

        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            if target.numel() == 0:
                loss = pred.sum() * 0.0
                return loss, 0

        lprobs = utils.log_softmax(pred, dim=-1)
        loss = F.nll_loss(
            lprobs,
            target.long(),
            ignore_index=self._get_contentvec_pad_idx(),
            reduction="sum",
        )
        sample_size = target.numel()
        return loss, sample_size

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        contentvec_loss_sum = sum(
            log.get("contentvec_loss", 0) for log in logging_outputs
        )
        contentvec_sample_size = sum(
            log.get("contentvec_sample_size", 0) for log in logging_outputs
        )
        if contentvec_sample_size > 0:
            metrics.log_scalar(
                "contentvec_loss",
                contentvec_loss_sum / contentvec_sample_size,
                contentvec_sample_size,
                round=5,
            )

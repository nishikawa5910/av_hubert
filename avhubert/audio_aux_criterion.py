# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict

import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)


@dataclass
class LabelSmoothedCrossEntropyWithAudioConfig(LabelSmoothedCrossEntropyCriterionConfig):
    audio_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for auxiliary audio label loss"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_audio",
    dataclass=LabelSmoothedCrossEntropyWithAudioConfig,
)
class LabelSmoothedCrossEntropyWithAudioCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, audio_loss_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.audio_loss_weight = audio_loss_weight
        self.audio_pad = None
        if getattr(task, "dictionaries", None) is not None and len(task.dictionaries) > 1:
            self.audio_pad = task.dictionaries[1].pad()

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

        audio_logits = None
        if isinstance(net_output, tuple) and len(net_output) > 1 and isinstance(net_output[1], dict):
            audio_logits = net_output[1].get("audio_logits")

        if audio_logits is not None and "audio_targets" in sample:
            pad_idx = self.audio_pad if self.audio_pad is not None else self.padding_idx
            lprobs_audio = utils.log_softmax(audio_logits, dim=-1)
            target_audio = sample["audio_targets"]
            audio_loss = F.nll_loss(
                lprobs_audio.view(-1, lprobs_audio.size(-1)),
                target_audio.view(-1),
                ignore_index=pad_idx,
                reduction="sum",
            )
            loss = loss + self.audio_loss_weight * audio_loss
            logging_output["audio_loss"] = audio_loss.data
            logging_output["audio_ntokens"] = sample.get("audio_ntokens", 0)

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = n_correct
            logging_output["total"] = total

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

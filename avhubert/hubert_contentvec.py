# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoderDecoderModel, register_model

from .decoder import TransformerDecoder
from .hubert_asr import AVHubertAsrConfig, AVHubertSeq2SeqConfig, HubertEncoder, HubertEncoderWrapper, Linear, Embedding

logger = logging.getLogger(__name__)


class ContentVecDecoder(nn.Module):
    def __init__(self, input_dim: int, cfg) -> None:
        super().__init__()
        self.output_dim = (
            cfg.contentvec_num_classes
            if cfg.contentvec_target_type == "class"
            else cfg.contentvec_dim
        )
        if cfg.contentvec_target_type == "class" and self.output_dim <= 0:
            raise ValueError("contentvec_num_classes must be > 0 for class targets")
        self.proj_in = (
            Linear(input_dim, cfg.contentvec_decoder_dim, bias=False)
            if input_dim != cfg.contentvec_decoder_dim
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.contentvec_decoder_dim,
            nhead=cfg.contentvec_decoder_attention_heads,
            dim_feedforward=cfg.contentvec_decoder_ffn_dim,
            dropout=cfg.contentvec_decoder_dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.contentvec_decoder_layers
        )
        self.proj_out = Linear(cfg.contentvec_decoder_dim, self.output_dim)

    def forward(self, x, padding_mask=None):
        if self.proj_in is not None:
            x = self.proj_in(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.proj_out(x)


@dataclass
class AVHubertContentVecConfig(AVHubertAsrConfig):
    contentvec_target_type: str = field(
        default="float",
        metadata={"help": "contentvec target type: float or class"},
    )
    contentvec_num_classes: int = field(
        default=0,
        metadata={"help": "number of classes for class targets"},
    )
    contentvec_dim: int = field(
        default=768, metadata={"help": "contentvec feature dimension"}
    )
    contentvec_decoder_dim: int = field(
        default=768, metadata={"help": "contentvec decoder hidden dimension"}
    )
    contentvec_decoder_layers: int = field(
        default=2, metadata={"help": "number of contentvec decoder layers"}
    )
    contentvec_decoder_attention_heads: int = field(
        default=4, metadata={"help": "attention heads in contentvec decoder"}
    )
    contentvec_decoder_ffn_dim: int = field(
        default=3072, metadata={"help": "FFN dimension in contentvec decoder"}
    )
    contentvec_decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout in contentvec decoder"}
    )


@register_model("av_hubert_contentvec", dataclass=AVHubertContentVecConfig)
class AVHubertContentVecModel(BaseFairseqModel):
    def __init__(self, cfg: AVHubertContentVecConfig, encoder: HubertEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = ContentVecDecoder(encoder.encoder_output_dim, cfg)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: AVHubertContentVecConfig, task):
        encoder = HubertEncoder(cfg, tgt_dict=None)
        return cls(cfg, encoder)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.encoder(**kwargs)
        pred = self.decoder(
            encoder_out["encoder_out"], encoder_out.get("encoder_padding_mask")
        )
        return {
            "pred": pred,  # T x B x C
            "padding_mask": encoder_out.get("encoder_padding_mask"),
        }


@dataclass
class AVHubertContentVecSeq2SeqConfig(AVHubertSeq2SeqConfig):
    contentvec_target_type: str = field(
        default="float",
        metadata={"help": "contentvec target type: float or class"},
    )
    contentvec_num_classes: int = field(
        default=0,
        metadata={"help": "number of classes for class targets"},
    )
    contentvec_dim: int = field(
        default=768, metadata={"help": "contentvec feature dimension"}
    )
    contentvec_decoder_dim: int = field(
        default=768, metadata={"help": "contentvec decoder hidden dimension"}
    )
    contentvec_decoder_layers: int = field(
        default=2, metadata={"help": "number of contentvec decoder layers"}
    )
    contentvec_decoder_attention_heads: int = field(
        default=4, metadata={"help": "attention heads in contentvec decoder"}
    )
    contentvec_decoder_ffn_dim: int = field(
        default=3072, metadata={"help": "FFN dimension in contentvec decoder"}
    )
    contentvec_decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout in contentvec decoder"}
    )
    contentvec_decoder_path: Optional[str] = field(
        default=None, metadata={"help": "checkpoint path for contentvec decoder"}
    )
    freeze_contentvec_decoder: bool = field(
        default=True, metadata={"help": "freeze contentvec decoder weights"}
    )


@register_model("av_hubert_contentvec_seq2seq", dataclass=AVHubertContentVecSeq2SeqConfig)
class AVHubertContentVecSeq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, contentvec_decoder, fusion_proj, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.contentvec_decoder = contentvec_decoder
        self.fusion_proj = fusion_proj
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: AVHubertContentVecSeq2SeqConfig, task):
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state["task_state"])

        encoder_ = task_pretrain.build_model(w2v_args.model)
        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            del state["model"]["mask_emb"]
            encoder.w2v_model.load_state_dict(state["model"], strict=False)
        encoder.w2v_model.remove_pretraining_modules()

        decoder_embed_tokens = Embedding(len(task.target_dictionary), cfg.decoder_embed_dim, task.target_dictionary.pad())
        decoder = TransformerDecoder(cfg, task.target_dictionary, decoder_embed_tokens)

        encoder_dim = encoder.w2v_model.encoder_embed_dim
        contentvec_decoder = ContentVecDecoder(encoder_dim, cfg)
        if cfg.contentvec_decoder_path is not None:
            cv_state = checkpoint_utils.load_checkpoint_to_cpu(cfg.contentvec_decoder_path)
            cv_model_state = cv_state.get("model", {})
            decoder_state = {
                key.replace("decoder.", ""): value
                for key, value in cv_model_state.items()
                if key.startswith("decoder.")
            }
            missing, unexpected = contentvec_decoder.load_state_dict(decoder_state, strict=False)
            if missing:
                logger.warning("Missing contentvec decoder keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected contentvec decoder keys: %s", unexpected)

        contentvec_dim = (
            cfg.contentvec_num_classes
            if cfg.contentvec_target_type == "class"
            else cfg.contentvec_dim
        )
        fusion_proj = Linear(
            encoder_dim + contentvec_dim,
            cfg.decoder_embed_dim,
        )
        return cls(encoder, decoder, contentvec_decoder, fusion_proj, cfg)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.encoder(**kwargs)
        padding_mask = encoder_out.get("encoder_padding_mask")
        if self.cfg.freeze_contentvec_decoder:
            with torch.no_grad():
                contentvec_pred = self.contentvec_decoder(
                    encoder_out["encoder_out"], padding_mask
                )
        else:
            contentvec_pred = self.contentvec_decoder(
                encoder_out["encoder_out"], padding_mask
            )
        fused = torch.cat([encoder_out["encoder_out"], contentvec_pred], dim=-1)
        fused = self.fusion_proj(fused)
        fused_out = {
            "encoder_out": fused,
            "encoder_padding_mask": padding_mask,
        }
        decoder_out = self.decoder(
            prev_output_tokens=kwargs["prev_output_tokens"],
            encoder_out=fused_out,
        )
        if isinstance(decoder_out, tuple):
            if len(decoder_out) > 1 and isinstance(decoder_out[1], dict):
                decoder_out[1]["contentvec_pred"] = contentvec_pred
                decoder_out[1]["contentvec_padding_mask"] = padding_mask
            else:
                decoder_out = (
                    decoder_out[0],
                    {
                        "contentvec_pred": contentvec_pred,
                        "contentvec_padding_mask": padding_mask,
                    },
                )
        else:
            decoder_out = (
                decoder_out,
                {
                    "contentvec_pred": contentvec_pred,
                    "contentvec_padding_mask": padding_mask,
                },
            )
        return decoder_out

#!/usr/bin/env bash
set -euo pipefail

# Seq2seq training with AVHubERT encoder + contentvec km500 decoder outputs,
# jointly optimizing s2s and contentvec losses.
# Replace /path/to/... with your actual paths.

data_path="/path/to/data"
label_dir="/path/to/labels"
spm_model="/path/to/wiki-ja.model"
pretrained_w2v="/path/to/avhubert_pretrained.pt"
contentvec_decoder="/path/to/contentvec_decoder_km500.pt"
run_dir="/path/to/exp/contentvec_km500_seq2seq_joint_vox_433h"

fairseq-hydra-train \
  --config-dir "$(pwd)/avhubert/conf/finetune" \
  --config-name contentvec_km500_seq2seq_joint_base_vox_433h \
  common.user_dir="$(pwd)/avhubert" \
  task.data="$data_path" \
  task.label_dir="$label_dir" \
  task.tokenizer_bpe_model="$spm_model" \
  model.w2v_path="$pretrained_w2v" \
  model.contentvec_decoder_path="$contentvec_decoder" \
  hydra.run.dir="$run_dir"

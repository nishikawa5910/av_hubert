#!/usr/bin/env bash
set -euo pipefail

# 500-class kmeans (km) fine-tuning example
# Replace /path/to/... with your actual paths.

data_path="/path/to/data"
label_dir="/path/to/labels"
pretrained_w2v="/path/to/avhubert_pretrained.pt"
run_dir="/path/to/exp/km500_vox_433h"

fairseq-hydra-train \
  --config-dir "$(pwd)/avhubert/conf/finetune" \
  --config-name km500_base_vox_433h \
  common.user_dir="$(pwd)/avhubert" \
  task.data="$data_path" \
  task.label_dir="$label_dir" \
  model.w2v_path="$pretrained_w2v" \
  hydra.run.dir="$run_dir"

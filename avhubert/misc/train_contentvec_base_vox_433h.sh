#!/usr/bin/env bash
set -euo pipefail

# contentvec_base_vox_433h.yaml を使って contentvec デコーダのみ学習する例
# 事前に以下のパスを自分の環境に合わせて更新してください。

data_path="/path/to/data"
label_dir="/path/to/data"
user_dir="/path/to/avhubert"
config_dir="${user_dir}/conf/finetune"
config_name="contentvec_base_vox_433h.yaml"
pretrained_w2v="/path/to/base_vox_iter5.pt"
run_dir="/path/to/finetunedmodel/contentvec_base_vox_433h"

command=(
  fairseq-hydra-train
  --config-dir "${config_dir}"
  --config-name "${config_name}"
  "task.data=${data_path}"
  "task.label_dir=${label_dir}"
  "model.w2v_path=${pretrained_w2v}"
  "hydra.run.dir=${run_dir}"
  "common.user_dir=${user_dir}"
)

printf 'Running: %q ' "${command[@]}"
printf '\n'
"${command[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Example script for training the contentvec decoder only.
# Fill in the paths below before running.

data_path="/path/to/data"
label_dir="/path/to/data"
user_dir="/path/to/avhubert"
config_dir="${user_dir}/conf/finetune"
config_name="base_vox_433h.yaml"
spm_model="/path/to/wiki-ja.model"
pretrained_w2v="/path/to/base_vox_iter5.pt"
run_dir="/path/to/finetunedmodel/model0"

command=(
  fairseq-hydra-train
  --config-dir "${config_dir}"
  --config-name "${config_name}"
  "task.data=${data_path}"
  "task.label_dir=${label_dir}"
  "task.labels=[contentvec]"
  "task.label_types=[float]"
  "task.label_rate=50"
  "task.is_s2s=false"
  "task.single_target=true"
  "task.fine_tuning=true"
  "task.tokenizer_bpe_model=${spm_model}"
  "model._name=av_hubert_contentvec"
  "model.w2v_path=${pretrained_w2v}"
  "model.freeze_finetune_updates=999999"
  "criterion._name=contentvec_mse"
  "hydra.run.dir=${run_dir}"
  "common.user_dir=${user_dir}"
)

printf 'Running: %q ' "${command[@]}"
printf '\n'
"${command[@]}"

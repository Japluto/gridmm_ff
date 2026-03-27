#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_NAV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MAP_NAV_DIR}/.." && pwd)"
cd "${MAP_NAV_DIR}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
train_alg="${TRAIN_ALG:-dagger}"
features="${FEATURES:-vitbase}"
ft_dim="${FT_DIM:-768}"
obj_features="${OBJ_FEATURES:-butd}"
obj_ft_dim="${OBJ_FT_DIM:-2048}"
ngpus="${NGPUS:-1}"
seed="${SEED:-0}"
mode="${1:-test}"

name="Grid_Map-${train_alg}-${features}-soon-single-gpu"
name="${name}-seed.${seed}"
outdir="${OUTDIR:-${DATA_ROOT}/SOON/exprs_map/eval/${name}}"

default_resume="${DATA_ROOT}/trained_models/soon_best"
resume_file="${RESUME_FILE:-${default_resume}}"

flag="--root_dir ${DATA_ROOT}
      --dataset soon
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert
      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints
      --dagger_sample sample
      --train_alg ${train_alg}
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100
      --batch_size 1
      --lr 5e-5
      --iters 10000
      --log_every 100
      --optim adamW
      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}
      --ml_weight 0.2
      --feat_dropout 0.4
      --dropout 0.5
      --gamma 0."

required_paths=(
  "${DATA_ROOT}/SOON/annotations/bert_enc/test_enc.jsonl"
  "${DATA_ROOT}/SOON/annotations/bert_enc/test_v2_enc.jsonl"
  "${DATA_ROOT}/R2R/connectivity/scans.txt"
  "${DATA_ROOT}/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
  "${DATA_ROOT}/R2R/features/depth.hdf5"
  "${DATA_ROOT}/R2R/features/clip_p32.hdf5"
  "${DATA_ROOT}/R2R/features/viewpoint_info.json"
  "${DATA_ROOT}/SOON/features/filtered_butd_bboxes.hdf5"
  "${resume_file}"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ "${ngpus}" -eq 1 ]]; then
  launcher=(python)
else
  if command -v torchrun >/dev/null 2>&1; then
    launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${ngpus}")
  else
    launcher=(python -m torch.distributed.launch --nproc_per_node="${ngpus}")
  fi
fi

cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

case "${mode}" in
  train)
    echo "Running SOON training on ${cuda_devices}, output: ${outdir}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main.py ${flag} \
      --resume_file "${resume_file}" \
      --eval_first
    ;;
  test)
    echo "Running SOON test on ${cuda_devices}, output: ${outdir}, checkpoint: ${resume_file}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main.py ${flag} \
      --test --submit \
      --resume_file "${resume_file}"
    ;;
  *)
    echo "Usage: bash scripts/run_soon.sh [test|train]" >&2
    exit 1
    ;;
esac

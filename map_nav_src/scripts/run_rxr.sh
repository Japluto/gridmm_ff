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
ngpus="${NGPUS:-1}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-1}"
mode="${1:-train}"
dynamic_memory_mode="${DYNAMIC_MEMORY_MODE:-off}"
dynamic_memory_extra_args="${DYNAMIC_MEMORY_EXTRA_ARGS:-}"

name="Grid_Map-${train_alg}-${features}-single-gpu-rxr"
name="${name}-seed.${seed}"
outdir="${OUTDIR:-${DATA_ROOT}/RXR/exprs_map/finetune/${name}}"

pretrain_ckpt="${PRETRAIN_CKPT:-}"
resume_file="${RESUME_FILE:-}"

dynamic_memory_args=()
case "${dynamic_memory_mode}" in
  off)
    dynamic_memory_args+=(--dynamic_memory_mode off)
    ;;
  update_only)
    dynamic_memory_args+=(--dynamic_memory_enabled --dynamic_memory_mode update_only)
    ;;
  decay_only)
    dynamic_memory_args+=(--dynamic_memory_enabled --dynamic_memory_mode decay_only --dynamic_memory_decay_enabled)
    ;;
  full)
    dynamic_memory_args+=(--dynamic_memory_enabled --dynamic_memory_mode full --dynamic_memory_decay_enabled)
    ;;
  *)
    echo "Unsupported DYNAMIC_MEMORY_MODE: ${dynamic_memory_mode}" >&2
    echo "Use one of: off, update_only, decay_only, full" >&2
    exit 1
    ;;
esac

if [[ -n "${dynamic_memory_extra_args}" ]]; then
  # shellcheck disable=SC2206
  extra_dynamic_memory_args=(${dynamic_memory_extra_args})
  dynamic_memory_args+=("${extra_dynamic_memory_args[@]}")
fi

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer xlm

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 20
      --max_instr_len 250

      --batch_size ${batch_size}
      --lr 1e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0."

required_paths=(
  "${DATA_ROOT}/RXR/annotations/rxr_train_enc.jsonl"
  "${DATA_ROOT}/RXR/annotations/rxr_val_seen_enc.jsonl"
  "${DATA_ROOT}/RXR/annotations/rxr_val_unseen_enc.jsonl"
  "${DATA_ROOT}/R2R/connectivity/scans.txt"
  "${DATA_ROOT}/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
  "${DATA_ROOT}/R2R/features/depth.hdf5"
  "${DATA_ROOT}/R2R/features/clip_p32.hdf5"
  "${DATA_ROOT}/R2R/features/viewpoint_info.json"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${DATA_ROOT}/Matterport3D/v1_unzip_scans" ]]; then
  echo "Warning: ${DATA_ROOT}/Matterport3D/v1_unzip_scans not found." >&2
  echo "The discrete RxR code may still run with precomputed features, but if MatterSim complains about scan data, place MP3D scans there." >&2
fi

if (( ngpus > 1 )); then
  if command -v torchrun >/dev/null 2>&1; then
    launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${ngpus}")
  else
    launcher=(python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node="${ngpus}")
  fi
else
  launcher=(python3)
fi

cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

case "${mode}" in
  train)
    extra_args=()
    if [[ -n "${pretrain_ckpt}" ]]; then
      if [[ ! -e "${pretrain_ckpt}" ]]; then
        echo "Missing PRETRAIN_CKPT: ${pretrain_ckpt}" >&2
        exit 1
      fi
      extra_args+=(--bert_ckpt_file "${pretrain_ckpt}")
    fi
    echo "Running RxR training on ${cuda_devices}, output: ${outdir}, batch_size=${batch_size}, dynamic_memory=${dynamic_memory_mode}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main_rxr.py ${flag} "${dynamic_memory_args[@]}" "${extra_args[@]}"
    ;;
  test)
    if [[ -z "${resume_file}" ]]; then
      echo "RESUME_FILE is required for test mode." >&2
      exit 1
    fi
    if [[ ! -e "${resume_file}" ]]; then
      echo "Missing RESUME_FILE: ${resume_file}" >&2
      exit 1
    fi
    echo "Running RxR test on ${cuda_devices}, output: ${outdir}, checkpoint: ${resume_file}, batch_size=${batch_size}, dynamic_memory=${dynamic_memory_mode}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main_rxr.py ${flag} \
      "${dynamic_memory_args[@]}" \
      --test --submit \
      --resume_file "${resume_file}"
    ;;
  *)
    echo "Usage: bash scripts/run_rxr.sh [train|test]" >&2
    exit 1
    ;;
esac

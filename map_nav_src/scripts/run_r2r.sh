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
mode="${1:-test}"
dynamic_memory_mode="${DYNAMIC_MEMORY_MODE:-off}"
dynamic_memory_extra_args="${DYNAMIC_MEMORY_EXTRA_ARGS:-}"
dual_stop_mode="${DUAL_STOP_MODE:-off}"
dual_stop_extra_args="${DUAL_STOP_EXTRA_ARGS:-}"
instr_aug_mode="${INSTR_AUG_MODE:-off}"
instr_aug_extra_args="${INSTR_AUG_EXTRA_ARGS:-}"

name="Grid_Map-${train_alg}-${features}-single-gpu"
name="${name}-seed.${seed}"
outdir="${OUTDIR:-${DATA_ROOT}/R2R/exprs_map/eval/${name}}"

default_resume="${DATA_ROOT}/trained_models/r2r_best"
resume_file="${RESUME_FILE:-${default_resume}}"

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

dual_stop_args=()
case "${dual_stop_mode}" in
  off)
    ;;
  on)
    dual_stop_args+=(--dual_stop_enabled)
    ;;
  *)
    echo "Unsupported DUAL_STOP_MODE: ${dual_stop_mode}" >&2
    echo "Use one of: off, on" >&2
    exit 1
    ;;
esac

if [[ -n "${dual_stop_extra_args}" ]]; then
  # shellcheck disable=SC2206
  extra_dual_stop_args=(${dual_stop_extra_args})
  dual_stop_args+=("${extra_dual_stop_args[@]}")
fi

instr_aug_args=()
case "${instr_aug_mode}" in
  off)
    ;;
  on)
    instr_aug_args+=(--instr_aug_enabled)
    ;;
  *)
    echo "Unsupported INSTR_AUG_MODE: ${instr_aug_mode}" >&2
    echo "Use one of: off, on" >&2
    exit 1
    ;;
esac

if [[ -n "${instr_aug_extra_args}" ]]; then
  # shellcheck disable=SC2206
  extra_instr_aug_args=(${instr_aug_extra_args})
  instr_aug_args+=("${extra_instr_aug_args[@]}")
fi

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 4
      --lr 1e-5
      --iters 20000
      --log_every 500
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

required_paths=(
  "${DATA_ROOT}/R2R/annotations/R2R_test_enc.json"
  "${DATA_ROOT}/R2R/connectivity/scans.txt"
  "${DATA_ROOT}/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
  "${DATA_ROOT}/R2R/features/depth.hdf5"
  "${DATA_ROOT}/R2R/features/clip_p32.hdf5"
  "${DATA_ROOT}/R2R/features/viewpoint_info.json"
  "${resume_file}"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

if command -v torchrun >/dev/null 2>&1; then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${ngpus}")
else
  launcher=(python3 -m torch.distributed.launch --master_port 29502 --nproc_per_node="${ngpus}")
fi

cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

# 这是一个case语句，根据mode变量的值执行不同的操作
case "${mode}" in
  # 当mode为"train"时的处理分支
  train)
    echo "Running R2R training on ${cuda_devices}, output: ${outdir}, dynamic_memory=${dynamic_memory_mode}, dual_stop=${dual_stop_mode}, instr_aug=${instr_aug_mode}"
    # 设置CUDA设备并使用指定的启动器运行main_nav.py进行训练
    # --resume_file 指定恢复训练的文件
    # --eval_first 表示在开始训练前先进行评估
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main_nav.py ${flag} \
      "${dynamic_memory_args[@]}" \
      "${dual_stop_args[@]}" \
      "${instr_aug_args[@]}" \
      --resume_file "${resume_file}" \
      --eval_first
    ;;
  # 当mode为"test"时的处理分支
  test)
    echo "Running R2R test on ${cuda_devices}, output: ${outdir}, checkpoint: ${resume_file}, dynamic_memory=${dynamic_memory_mode}, dual_stop=${dual_stop_mode}, instr_aug=${instr_aug_mode}"
    # 设置CUDA设备并使用指定的启动器运行main_nav.py进行测试
    # --test 表示测试模式
    # --submit 表示提交测试结果
    # --resume_file 指定测试使用的模型检查点文件
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${launcher[@]}" main_nav.py ${flag} \
      "${dynamic_memory_args[@]}" \
      "${dual_stop_args[@]}" \
      "${instr_aug_args[@]}" \
      --test --submit \
      --resume_file "${resume_file}"
    ;;
  # 当mode值不为"train"或"test"时的默认处理分支
  *)
    # 打印使用说明到标准错误流并退出程序
    echo "Usage: bash scripts/run_r2r.sh [test|train]" >&2
    exit 1
    ;;
esac

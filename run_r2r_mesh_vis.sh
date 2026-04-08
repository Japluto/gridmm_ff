#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_r2r_mesh_vis.sh [limit] [fps] [seed]
#
# Notes:
# - The exact local "bt022_ms1" prediction file is not present in this workspace.
# - This script uses the closest available local decay-only + anti-loop R2R run:
#   bt022_ms2 (same decay and penalty settings, anti_loop_min_step=2).

LIMIT="${1:-3}"
FPS="${2:-2}"
SEED="${3:-$(date +%s)}"

echo "[info] R2R mesh visualization seed: ${SEED}"

/home/japluto/anaconda3/bin/conda run -n gridmm \
  python /home/japluto/VLN/GridMM_ff/map_nav_src/scripts/graph_nav_movie.py \
  --dataset r2r \
  --preds /home/japluto/VLN/GridMM_ff/default/r2r_decay_antiloop_bt022_ms2/preds/submit_val_train_seen.json \
  --annotations /home/japluto/VLN/GridMM_ff/datasets/R2R/annotations/R2R_val_train_seen_enc.json \
  --connectivity_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/connectivity \
  --mesh_dir /home/japluto/VLN/GridMM/VLN_CE/data/scene_datasets/mp3d \
  --output_dir /home/japluto/VLN/GridMM_ff/visualizations/mesh_bev_textured/r2r \
  --limit "${LIMIT}" \
  --fps "${FPS}" \
  --seed "${SEED}"

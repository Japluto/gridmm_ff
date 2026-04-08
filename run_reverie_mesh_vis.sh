#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_reverie_mesh_vis.sh [limit] [fps] [seed]

LIMIT="${1:-3}"
FPS="${2:-2}"
SEED="${3:-$(date +%s)}"

echo "[info] REVERIE mesh visualization seed: ${SEED}"

/home/japluto/anaconda3/bin/conda run -n gridmm \
  python /home/japluto/VLN/GridMM_ff/map_nav_src/scripts/graph_nav_movie.py \
  --dataset reverie \
  --preds /home/japluto/VLN/GridMM_ff/datasets/REVERIE/exprs_map/eval/Grid_Map-dagger-vitbase-reverie-single-gpu-seed.0/preds/final_decay_only_antiloop/submit_val_unseen_dynamic.json \
  --annotations /home/japluto/VLN/GridMM_ff/datasets/REVERIE/annotations/REVERIE_val_unseen_enc.json \
  --connectivity_dir /home/japluto/VLN/GridMM_ff/datasets/R2R/connectivity \
  --bbox_json /home/japluto/VLN/GridMM_ff/datasets/REVERIE/annotations/BBoxes.json \
  --mesh_dir /home/japluto/VLN/GridMM/VLN_CE/data/scene_datasets/mp3d \
  --output_dir /home/japluto/VLN/GridMM_ff/visualizations/mesh_bev_textured/reverie \
  --limit "${LIMIT}" \
  --fps "${FPS}" \
  --seed "${SEED}"

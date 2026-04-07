# DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation

This repository is a research fork of GridMM for our discrete VLN project:

**DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation**

DART-VLN focuses on lightweight, training-free test-time control for discrete vision-language navigation. Instead of adding new learnable modules, it improves inference behavior through two simple mechanisms:

- **Memory Decay**: softly downweights stale or redundant grid memory at readout time
- **Anti-Loop Regularization**: suppresses immediate backtracking during action selection

The current public-facing focus of this repository is:

- `R2R`
- `REVERIE`

## Preview

The repository includes a mesh-based bird's-eye trajectory renderer. The preview below is generated from the local visualization pipeline:

![Trajectory Preview](trajectory.gif)

## Highlights

- **Training-free inference enhancement**: no retraining and no additional learnable parameters
- **Clean ablations**: easy to switch between `off`, `decay_only`, `update_only`, and `full`
- **Behavior-aware decoding**: immediate backtrack suppression improves trajectory efficiency
- **Textured bird's-eye visualization**: real Matterport mesh rendering with topological graph overlay
- **Discrete VLN support**: organized around practical evaluation on `R2R` and `REVERIE`

## Repository Overview

`GridMM_ff` is a working discrete-navigation fork extracted from the original GridMM codebase. It keeps the discrete VLN foundation while shifting the focus to test-time improvements and qualitative analysis.

Main directories:

- `map_nav_src/`: discrete navigation agents, environments, evaluation, and visualization
- `pretrain_src/`: pretraining code kept for completeness
- `preprocess/`: preprocessing utilities

Large datasets, precomputed features, and checkpoints are expected to exist locally and are usually git-ignored.

## Method Summary

### Memory Decay

The memory-decay module applies a lightweight read-side reweighting rule to grid memory:

- stale memory receives lower weight
- repeated or redundant memory is softly suppressed
- the backbone and learned parameters remain unchanged

In our current experiments, **`decay_only` is the most reliable memory-side setting**.

### Anti-Loop Regularization

The anti-loop module adds a small test-time penalty to candidate actions that would immediately return to the previous viewpoint.

In practice, the most useful behavior is:

- **immediate backtrack suppression**

This makes the agent less likely to waste steps in short local loops while remaining lightweight and easy to ablate.

## Recommended Workflow

The practical workflow in this repository is:

1. Prepare the discrete VLN environment and local datasets.
2. Run evaluation with `decay_only` or `decay_only + anti-loop`.
3. Export prediction files for `R2R` or `REVERIE`.
4. Render trajectory visualizations on top of a textured bird's-eye mesh.

Current rule of thumb:

- use **`decay_only`** for the cleanest memory-side gain
- use **`decay_only + anti-loop`** for the strongest behavior / efficiency trade-off

## Running Experiments

The main experiment entry points are:

- `map_nav_src/scripts/run_r2r.sh`
- `map_nav_src/scripts/run_reverie.sh`

Typical examples:

```bash
cd map_nav_src
bash scripts/run_r2r.sh test
```

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only bash scripts/run_r2r.sh test
```

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
ANTI_LOOP_MODE=on \
bash scripts/run_r2r.sh test
```

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
ANTI_LOOP_MODE=on \
bash scripts/run_reverie.sh test
```

## Explicit Test-Time Configurations

The current tuned values are:

- `dynamic_memory_decay_lambda = 0.12`
- `dynamic_memory_min_mem_weight = 0.35`
- `dynamic_memory_max_mem_weight = 1.0`
- `anti_loop_backtrack_penalty = 0.22`
- `anti_loop_revisit_penalty = 0.0`
- `anti_loop_revisit_thresh = 2`
- `anti_loop_min_step = 1`

### R2R: decay-only

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=off \
bash scripts/run_r2r.sh test
```

### R2R: decay-only + anti-loop

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_r2r.sh test
```

### REVERIE: decay-only + anti-loop

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_reverie.sh test
```

## Command-Line Tuning

The scripts use environment variables to keep ablations clean and reproducible:

- `DYNAMIC_MEMORY_MODE=off|update_only|decay_only|full`
- `ANTI_LOOP_MODE=off|on`
- `DYNAMIC_MEMORY_EXTRA_ARGS="..."`
- `ANTI_LOOP_EXTRA_ARGS="..."`

Example:

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.08 --dynamic_memory_min_mem_weight 0.45" \
bash scripts/run_r2r.sh test
```

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.28 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_r2r.sh test
```

## Visualization

This repository includes mesh-based bird's-eye trajectory visualization for discrete navigation.

Main files:

- `map_nav_src/scripts/graph_nav_movie.py`
- `map_nav_src/scripts/decode_glb_basis_textures.js`
- `run_r2r_mesh_vis.sh`
- `run_reverie_mesh_vis.sh`

The visualization pipeline:

- uses the real Matterport mesh as the bird's-eye base layer
- projects the topological graph onto the mesh plane
- overlays the ground-truth path, predicted path, current viewpoint, and next step
- exports `frames/`, `trajectory.gif`, and `trajectory.mp4`

### Visualization Flow

1. Run evaluation and produce prediction files.
2. Ensure annotations, connectivity files, and `mp3d/*.glb` meshes are available locally.
3. Run the visualization script for `R2R` or `REVERIE`.
4. Inspect the generated GIF / MP4 and frame sequence.

### One-Command Visualization

From the repository root:

```bash
./run_r2r_mesh_vis.sh
./run_reverie_mesh_vis.sh
```

Both scripts also support optional arguments:

```bash
./run_r2r_mesh_vis.sh 5 3
./run_reverie_mesh_vis.sh 5 3
```

where the two arguments are:

- number of episodes to render
- output FPS

### Output Location

The current scripts write visualization results to:

- `visualizations/mesh_bev_textured/r2r`
- `visualizations/mesh_bev_textured/reverie`

Each rendered episode contains:

- `frames/frame_000.png`, `frame_001.png`, ...
- `trajectory.gif`
- `trajectory.mp4`

### Notes

- The first run may be slower because texture decoder dependencies are initialized.
- The renderer uses embedded MP3D texture data when available.
- If texture decoding fails, it falls back to a simpler occupancy-style mesh overlay.

## Acknowledgments

This repository is built on top of the original **GridMM** codebase and extends it with lightweight test-time control and trajectory visualization for discrete VLN.

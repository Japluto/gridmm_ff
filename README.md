# DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation

This repository is a research fork of GridMM(nice work) for our discrete VLN project:

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

## Effective Repository Structure

`GridMM_ff` is a working discrete-navigation fork extracted from the original GridMM codebase. It keeps the discrete VLN foundation while shifting the focus to test-time improvements and qualitative analysis.

```text
GridMM_ff/
├── map_nav_src/                  # active navigation code
│   ├── r2r/                      # R2R task logic
│   ├── reverie/                  # REVERIE task logic
│   ├── scripts/                  # train / eval / visualization entrypoints
│   ├── models/                   # navigation models
│   └── utils/                    # shared utilities
├── preprocess/                   # preprocessing helpers
├── pretrain_src/                 # retained pretraining code
├── datasets/                     # local datasets and features (git-ignored)
│   ├── R2R/
│   ├── REVERIE/
│   └── Matterport3D/
├── default/                      # local runs, checkpoints, preds (git-ignored)
├── visualizations/               # scratch visualization outputs (git-ignored)
├── example_831_0/                # curated example package
├── matterport_download/          # local Matterport raw downloads
├── matterport_preview/           # local pano preview assets
├── run_r2r_mesh_vis.sh
└── run_reverie_mesh_vis.sh
```

The parts that matter most for day-to-day work are:

- `map_nav_src/`: experiments, evaluation, and rendering
- `datasets/`: local annotations, connectivity, features, and simulator assets
- `default/`: local checkpoints / logs / predictions
- `example_831_0/`: a packaged public example for inspection and demos

Large datasets, precomputed features, and checkpoints are local-only resources and are intentionally git-ignored.

## Example Package: `example_831_0`

`example_831_0/` is a slim public example for:

- `scan = JeFG25nYj2p`
- `instr_id = 831_0`

The public branch keeps the final preview assets only, so the repository stays lightweight enough to push and clone comfortably.

Included files:

- `latest_bev_topo.png`: final recommended static `topo + BEV` image
- `latest_graph_nav_frame.png`: final recommended single frame in graph-nav layout
- `latest_bev_trajectory.gif` / `latest_bev_trajectory.mp4`: final recommended animated result
- `agent_fpv_trajectory.gif` / `agent_fpv_trajectory.mp4`: first-person preview
- `agent_fpv_contact_sheet.png`: compact FPV summary image

The full intermediate rendering history stays local and is not part of the public branch.

See `example_831_0/README.md` for the file-level description of the public example package.

## Data Setup

This repository expects datasets to be prepared locally. We keep the code in git, but not the datasets, features, or model weights.

### 1. R2R

Official R2R task data is released with Matterport3DSimulator. The most useful official entry point is the R2R task README, which points to the standard task format and the `tasks/R2R/data/download.sh` helper.

In this fork, the local targets are typically:

- `datasets/R2R/annotations/`
- `datasets/R2R/connectivity/`
- `datasets/R2R/features/`

Common files used by this repo include:

- `datasets/R2R/annotations/R2R_train_enc.json`
- `datasets/R2R/annotations/R2R_val_seen_enc.json`
- `datasets/R2R/annotations/R2R_val_unseen_enc.json`
- `datasets/R2R/connectivity/scans.txt`
- `datasets/R2R/connectivity/*_connectivity.json`

### 2. REVERIE

REVERIE data is maintained in the official REVERIE repository. Use the official repo as the reference for task data layout and object-grounding annotations.

In this fork, the local targets are typically:

- `datasets/REVERIE/annotations/`
- `datasets/REVERIE/features/`

The key files we use are:

- `datasets/REVERIE/annotations/REVERIE_*_enc.json`
- `datasets/REVERIE/annotations/BBoxes.json`
- `datasets/REVERIE/features/obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5`

### 3. Matterport3D

For simulator RGB views, pano export, and our visualization work, you need Matterport3D access and the official download script.

Useful official resources:

- download script: `http://kaldir.vc.cit.tum.de/matterport/download_mp.py`
- scan list: `http://kaldir.vc.cit.tum.de/matterport/v1/scans.txt`

At minimum, Matterport3DSimulator requires:

- `matterport_skybox_images`

For RGB + depth aligned workflows, download:

- `matterport_skybox_images`
- `undistorted_camera_parameters`
- `undistorted_depth_images`
- optionally `undistorted_color_images`

In our local setup, these assets usually appear in one of two places:

- `datasets/Matterport3D/v1_unzip_scans/` for simulator-oriented code paths
- `matterport_download/v1/scans/` for raw official downloads and FPV / BEV experiments

### 4. Features and Weights

This fork does not bundle pretrained weights or large feature files.

Common local-only locations are:

- `data/pretrained_models/`
- `datasets/pretrained/`
- `datasets/trained_models/`
- `datasets/R2R/features/`
- `datasets/REVERIE/features/`

For the exact files required by each task, the quickest reference is the required-file check inside:

- `map_nav_src/scripts/run_r2r.sh`
- `map_nav_src/scripts/run_reverie.sh`

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
- Curated, hand-selected outputs are kept in `example_831_0/`; `visualizations/` is treated as a local scratch area.
- The public branch keeps only a lightweight subset of preview assets. Full local rendering histories are intentionally excluded.

## Official Data References

- Matterport3DSimulator README: https://github.com/peteanderson80/Matterport3DSimulator
- R2R task README: https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/README.md
- REVERIE repository: https://github.com/YuankaiQi/REVERIE
- Matterport3D official download script: http://kaldir.vc.cit.tum.de/matterport/download_mp.py

## Acknowledgments

This repository is built on top of the original **GridMM** codebase and extends it with lightweight test-time control and trajectory visualization for discrete VLN.

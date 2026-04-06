# DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation
---
> WORK LEADER :Japluto

This repository is a discrete-environment research fork of GridMM for the paper:

**DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-and-Language Navigation**

The project focuses on **small, training-free test-time improvements** for discrete VLN.
Instead of introducing new learnable modules, DART-VLN studies how far we can push performance and behavior quality by modifying:

- **memory readout**: soft decay of stale grid memory
- **action regularization**: lightweight anti-loop / immediate backtrack suppression

The main target datasets are:

- `R2R`
- `REVERIE`
- `RxR`

---
## What This Repo Is

`GridMM_ff` is a working copy extracted from the original `GridMM` codebase, with the focus shifted from continuous-environment pipelines to **discrete navigation and evaluation**.

The repository is organized around:

- `map_nav_src/`: discrete navigation agents, environments, and scripts
- `pretrain_src/`: pretraining code kept for completeness
- `preprocess/`: preprocessing utilities

Large resources such as datasets and pretrained checkpoints are expected locally but are git-ignored.

---
## Main Idea

Our working hypothesis is simple:

- the grid memory should not keep stale information forever
- old or redundant information should be softly downweighted at readout
- the agent should avoid obvious short loops such as immediate backtracking

This leads to two practical test-time components:

### 1. Memory Decay

We add a lightweight **memory decay** mechanism for grid memory readout.

- no retraining
- no new trainable parameters
- easy ablation through config flags

The strongest result so far is that **`decay_only` is more stable than more aggressive memory rewriting schemes** such as `update_only` or `full`.

### 2. Anti-Loop Regularization

We add a small **anti-loop penalty** at action selection time.

In practice, the useful part is:

- **immediate backtrack suppression**

That is, if the next hop of a candidate action would directly return to the previous viewpoint, we apply a small penalty before final action selection.

This mechanism is:

- test-time only
- non-invasive
- compatible with `decay_only`

---
## What We Tried

Over the course of this project, we explored several lightweight directions:

- **dynamic memory update** with heuristic write gates
- **soft memory decay** during readout
- **dual STOP** rules for R2R
- **instruction-side augmentation**
- **instruction-aware reranking**
- **anti-loop / backtrack suppression**

Current takeaways:

- `decay_only` is the most reliable memory-side improvement
- `anti-loop` is useful mainly as **immediate backtrack suppression**
- `instruction`-side heuristics and `dual STOP` were not as robust in our current setting

---
## Current Status

### R2R

This is the main development and evaluation track.

Current practical recommendation:

- use `decay_only` as the main score-oriented configuration
- use `decay_only + anti-loop` as a behavior/efficiency-oriented variant

### REVERIE

The same dynamic-memory and anti-loop ideas have also been migrated to REVERIE.

Empirically, `decay_only + anti-loop` appears especially interesting for **unseen generalization**, even when gains are not uniform across all splits.

### RxR

The discrete evaluation path and scripts are organized, but usable finetuned checkpoints may still need to be prepared depending on the experiment.

---
## Important Design Principle

This project intentionally favors:

- **test-time heuristics**
- **small code footprint**
- **clear ablations**
- **no architecture rewrite**

In other words, DART-VLN is about improving discrete VLN behavior with simple, explainable mechanisms rather than adding new learned components.

---
## Running Experiments

The main script entry points are:

- `map_nav_src/scripts/run_r2r.sh`
- `map_nav_src/scripts/run_reverie.sh`
- `map_nav_src/scripts/run_rxr.sh`

These scripts already expose the relevant switches for:

- dynamic memory mode
- anti-loop mode
- dataset-specific evaluation

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

---
### Fully Explicit Commands

The repository defaults already encode the current tuned values for:

- `dynamic_memory_decay_lambda = 0.12`
- `dynamic_memory_min_mem_weight = 0.35`
- `dynamic_memory_max_mem_weight = 1.0`
- `anti_loop_backtrack_penalty = 0.22`
- `anti_loop_revisit_penalty = 0.0`
- `anti_loop_revisit_thresh = 2`
- `anti_loop_min_step = 1`

If you want to make every switch explicit on the command line, you can write them out as follows.

R2R, decay-only:

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=off \
bash scripts/run_r2r.sh test
```

R2R, decay-only + anti-loop:

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_r2r.sh test
```

REVERIE, decay-only + anti-loop:

```bash
cd map_nav_src
DYNAMIC_MEMORY_MODE=decay_only \
DYNAMIC_MEMORY_EXTRA_ARGS="--dynamic_memory_decay_lambda 0.12 --dynamic_memory_min_mem_weight 0.35 --dynamic_memory_max_mem_weight 1.0" \
ANTI_LOOP_MODE=on \
ANTI_LOOP_EXTRA_ARGS="--anti_loop_backtrack_penalty 0.22 --anti_loop_revisit_penalty 0.0 --anti_loop_revisit_thresh 2 --anti_loop_min_step 1" \
bash scripts/run_reverie.sh test
```

---
### How To Tune from the Command Line

The scripts use environment variables to keep ablations clean.

- `DYNAMIC_MEMORY_MODE=off|update_only|decay_only|full`
- `ANTI_LOOP_MODE=off|on`
- `DYNAMIC_MEMORY_EXTRA_ARGS="..."`
- `ANTI_LOOP_EXTRA_ARGS="..."`

Examples:

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

---
## Acknowledgement

This repository is built on top of the original **GridMM** codebase and keeps its discrete VLN foundation while focusing on new test-time regularization ideas for memory and action selection.

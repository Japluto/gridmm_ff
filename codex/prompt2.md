You are working in my GridMM_ff repository.

I want a **minimal, no-training dual STOP implementation** for discrete VLN / REVERIE / RxR evaluation.

## Goal
Implement the simplest possible **dual STOP** rule at test time, with:
- no retraining
- no new learnable module
- minimal code changes
- easy ablation
- works for my discrete datasets first, especially R2R / REVERIE / RxR

The idea is:
STOP should not be decided by the model's STOP score alone.
It should require **two signals**:

1. **semantic STOP signal**
   - the original model STOP score / STOP probability is strong enough

2. **geometric/topological STOP signal**
   - the agent is already in a "reasonable-to-stop" local state, using a simple rule-based signal

## Keep this implementation minimal
Do NOT design a complicated planner.
Do NOT add a new network.
Do NOT refactor large parts of the code.

This should be only a small test-time decision rule around STOP.

## Desired minimal dual STOP logic
Use the following simple version first:

### semantic condition
`semantic_stop_ok = stop_score >= stop_score_thresh`
or equivalent if the code uses STOP logit / probability / action score.

### geometric/topological condition
Use the easiest available signal from the existing codebase.
Preferred order:

1. current step >= `min_stop_step`
2. AND one of the following is true:
   - current viewpoint has already been revisited enough times
   - or the best non-stop candidate is weak / close to stop score
   - or there are no clearly good unvisited forward candidates

Use the **smallest code footprint** version that matches the repo’s existing variables.

### final STOP decision
Only allow STOP if:
`semantic_stop_ok and geometric_stop_ok`

Otherwise, suppress STOP and choose the best non-stop action.

## Important preference
I want the **simplest version first**.
So do not over-engineer this.

In particular:
- if there is already a visited viewpoint history, reuse it
- if there is already a STOP score and ranked candidate actions, reuse them
- do not build a new geometric module
- do not add dataset-specific complicated logic unless it is extremely easy

## Optional REVERIE-specific enhancement
If REVERIE already has an easy-to-access object confidence / grounding confidence signal in the action selection path,
you may optionally use it as an additional semantic support term.
But:
- keep the base dual STOP common across all datasets
- do not make REVERIE-specific logic mandatory for the first patch

## What to inspect first
Please inspect the code and identify:

1. where STOP is currently selected
2. where action candidates are scored / ranked
3. where visited viewpoint history is tracked
4. the smallest insertion point for a dual STOP rule

Likely relevant places may be around:
- policy / agent action selection
- rollout / evaluation step
- candidate ranking code
- history / visited viewpoint tracking

## Configs to add
Add a few minimal config flags only:

- `DUAL_STOP_ENABLED` (bool)
- `DUAL_STOP_SCORE_THRESH` (float, default conservative)
- `DUAL_STOP_MIN_STEP` (int, default small positive value)
- `DUAL_STOP_MARGIN_THRESH` (float, for stop vs best-move comparison)
- `DUAL_STOP_REVISIT_THRESH` (int, for viewpoint revisit count)

Do not add many more parameters unless absolutely necessary.

## Ablation modes
Support at least:
- off
- dual_stop_on

If easy, also support:
- semantic_only
- dual_stop_full

But keep the implementation small.

## Validation target
I care most about:
- SR improvement
- no obvious SPL collapse
- minimal regression risk

## Deliverables
Please do this in order:

### Step 1
Inspect the repo and report:
- exact file(s) and function(s) where STOP is chosen
- exact file(s) and function(s) where visited history is available
- your minimal patch plan

### Step 2
Implement the minimal dual STOP patch.

### Step 3
Add the config flags.

### Step 4
Run the lightest possible validation / smoke test.

### Step 5
Report:
- changed files
- implemented STOP rule
- default thresholds
- how to turn it on/off
- possible failure cases

## Very important
This is intentionally a **small rule-based patch**.
Prefer a boring, stable implementation over a clever but invasive one.

Start with Step 1 inspection only. Do not make broad edits before identifying the smallest insertion point.
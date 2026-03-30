Your insertion plan for the minimal R2R-only instruction-aware rerank is approved, but there are several implementation guards that are REQUIRED before coding.

Please incorporate all of the following into the patch design.

## Approved overall plan
Keep the current plan:
- R2R only
- modify only:
  - `map_nav_src/r2r/agent.py`
  - `map_nav_src/r2r/parser.py`
- use:
  - raw instruction text from `obs[i]['instruction']`
  - candidate headings from `obs[i]['candidate'][j]['heading']`
  - existing `nav_logits / nav_probs`
- insert the rerank after scores are available and before final `argmax`
- do not modify the text encoder
- do not modify instruction text
- do not modify STOP logic
- do not touch training

## REQUIRED implementation guards

### 1. Candidate index alignment must be verified explicitly
Before applying any boost, verify the exact mapping between:
- `obs[i]['candidate'][j]`
and
- the corresponding non-stop action index in `nav_probs` / `nav_logits`

Important:
- STOP is usually index 0
- non-stop candidate indices may be shifted by +1
- do NOT assume the mapping without checking

In your Step 1 report, explicitly state:
- whether candidate `j` maps to score index `j+1`, or another mapping

This is critical.

---

### 2. Direction buckets must use thresholds, not just sign
Do NOT use a naive rule like:
- heading < 0 => left
- heading > 0 => right
- heading == 0 => straight

Instead, use a small threshold band.

Use a conservative bucketization such as:
- `straight` if `abs(heading) <= heading_straight_thresh`
- `left` if `heading < -heading_straight_thresh`
- `right` if `heading > heading_straight_thresh`

Add a minimal config flag if needed:
- `--instr_rerank_heading_straight_thresh`

Suggested default:
- `instr_rerank_heading_straight_thresh = 0.35`

Keep it simple and conservative.

---

### 3. Uncertainty gate must be computed on non-stop candidates only
The rerank is intended only for non-stop action selection.

Therefore:
- when computing top1/top2 closeness for uncertainty
- ignore STOP
- compare only non-stop candidate scores

Do NOT let STOP score determine whether the rerank is activated.

In your report, explicitly state:
- how uncertainty is computed
- which indices are included/excluded

---

### 4. Direction cue extraction must be conservative
Only apply the prior when there is a clear, non-conflicting direction cue.

Minimal supported cues:
- left
- right
- straight

Rules:
- if exactly one clear direction cue exists -> use it
- if multiple conflicting cues exist (e.g. both left and right) -> disable prior
- if no clear direction cue exists -> disable prior

Be conservative.
Do NOT try to resolve complex multi-cue instructions in the first version.

---

### 5. The boost must remain extremely small
Use a very weak additive rerank only.

Suggested default:
- `instr_rerank_dir_boost = 0.02`

The rerank is only meant to gently nudge uncertain early-stage action choice.
It must not overpower the model.

---

### 6. Keep early-stage gating
Only apply rerank when:
- `t < instr_rerank_max_step`

Suggested default:
- `instr_rerank_max_step = 4`

Keep this as-is unless you find a strong reason to change it.

---

### 7. Add minimal behavior statistics
In addition to debug prints, add small run-level counters if easy.

At the end of evaluation, report counts such as:
- rerank_trigger_count
- rerank_action_changed_count
- left/right/straight cue counts
- conflict_disabled_count
- no_cue_disabled_count

Keep it lightweight.
This will help us understand whether the rerank is actually being used.

## Updated minimal parser flags
Please support these flags in R2R parser:

- `--instr_rerank_enabled`
- `--instr_rerank_uncertainty_thresh`
- `--instr_rerank_max_step`
- `--instr_rerank_dir_boost`
- `--instr_rerank_heading_straight_thresh`
- `--instr_rerank_debug_print`

Suggested defaults:
- `instr_rerank_enabled = False`
- `instr_rerank_uncertainty_thresh = 0.04`
- `instr_rerank_max_step = 4`
- `instr_rerank_dir_boost = 0.02`
- `instr_rerank_heading_straight_thresh = 0.35`
- `instr_rerank_debug_print = False`

## Deliverables
Please do this in order:

### Step 1
Inspect and confirm:
1. candidate-score index mapping
2. exact uncertainty computation on non-stop candidates only
3. exact heading bucketization rule
4. exact direction-cue extraction rule
5. smallest insertion point

### Step 2
Implement the patch with all required guards above.

### Step 3
Add parser flags.

### Step 4
Run the lightest possible validation.

### Step 5
Report:
- changed files
- candidate-score mapping
- uncertainty formula
- heading thresholds
- cue extraction logic
- default parameter values
- whether rerank changes top1 action in practice
- run-level rerank statistics if added

## Important
Keep the patch small, conservative, and boring.
Do not broaden scope.
Do not add room/landmark rerank in this first version unless it is literally trivial and does not increase risk.
Start with Step 1 inspection only.



# Coverage Metric Explainer

Standalone blogpost-style page explaining the **Coverage** benchmark metric from the ECCV 2026 SBM paper.

## Viewing

Open `index.html` in a browser. No build step required — all assets are pre-generated static files.

## Asset generation

All figures are produced by `generate_coverage_assets.py`. Run:

```bash
python generate_coverage_assets.py
```

### Dependencies

- Python 3.10+
- `matplotlib`, `numpy` (standard scientific stack)

### What gets generated

| Asset | Generator | External data needed? |
|---|---|---|
| `fig_01_coverage_breakdown.svg/.png` | `render_breakdown()` | No |
| `fig_02_item_score_curves.svg` | `render_item_score_curves()` | No |
| `fig_03_requirement_groups.svg` | `render_requirement_groups()` | No |
| `fig_04_masterbed_coverage_case.png` | `render_masterbed_case()` | Yes |
| `fig_05_laundry_complementarity_case.png` | `render_laundry_case()` | Yes |
| `fig_06_coverage_leaderboard.svg` | `render_coverage_leaderboard()` | Yes (JSON only) |

### External data

Figures 04 and 05 (the room case studies) require:

1. The **benchmark repo** at `ai-services/tools/benchmarks/lbm-ddep` — specifically the coverage/navigation metric code, the inventory spec config, and the heldout prediction outputs.
2. The **LBM vocab directory** for entity name normalization.

Figure 06 (the leaderboard) only needs the exported JSON at `results/paper_exports/camera_ready_metrics_export.json`.

If the benchmark imports are unavailable, the script skips the case-study figures gracefully and still generates figures 01–03 and 06.

### Manual pixel edits

Figures 04 and 05 have had post-generation pixel-level patches applied (via PIL/numpy) to fix text overflow issues in the score cards. If these figures are regenerated from scratch, the fixes in the generation code should prevent the overflow, but verify visually:

- **fig_04**: The "Counts" line wraps onto two lines via `add_plan_card_text()`. The "Nav" row was removed from `score_lines()`.
- **fig_05**: The bottom explanatory text uses manual line breaks and `y=0.18` positioning to stay inside the card boundary.

## The Coverage formula explained

**Coverage** = 100 × ( (Σ wᵢsᵢ + Σ wᵍsᵍ) / (Σ wᵢ + Σ wᵍ) − p_extra )

### 1. The item score: sᵢ

Each item category (bed, end table, dresser, TV...) gets a score between 0 and 1.

**sᵢ = 1 − clip(α·dᵢ + β·oᵢ, 0, 1)**

- **sᵢ** = how well the prediction satisfied this specific item. 1 means perfect, 0 means total failure.
- **clip(x, 0, 1)** = just clamps the value so it never goes below 0 or above 1. Safety guardrail.

Where:

**dᵢ = max(0, minᵢ − predᵢ) / max(1, minᵢ)**

- **predᵢ** = how many of this item the model predicted (e.g. predicted 0 beds)
- **minᵢ** = the minimum required count from the room spec (e.g. a master bedroom requires at least 1 bed)
- **max(0, minᵢ − predᵢ)** = how many are *missing*. If you predicted enough, this is 0. If you predicted fewer than required, this is the shortfall.
- **/ max(1, minᵢ)** = normalizes the shortfall as a *fraction* of what was required. Missing 1 out of 2 required end tables = 0.5 (half missing). The `max(1, ...)` in the denominator just prevents division by zero when minᵢ is 0.
- **dᵢ** = the "missing fraction". Ranges from 0 (nothing missing) to 1 (everything missing).

**oᵢ = max(0, predᵢ − maxᵢ) / max(1, maxᵢ)**

- Same logic but in reverse: how much did you *overshoot* the maximum allowed count.
- **maxᵢ** = the maximum allowed count (e.g. a master bedroom allows at most 1 bed)
- **oᵢ** = the "overfill fraction". If you predicted 3 beds but max is 1, that's (3−1)/1 = 2.0 overfill.

**α = 1.0, β = 0.5**

- These are the penalty weights. **Missing items hurt twice as much as extra items.** The intuition: a bedroom with no bed is worse than a bedroom with two beds. Both are wrong, but one is more wrong.
- α=1.0 means the missing fraction hits the score at full strength
- β=0.5 means the overfill fraction hits at half strength

**Putting it together for one item:** if you predicted exactly the right count, dᵢ=0 and oᵢ=0, so sᵢ = 1−0 = 1 (perfect). If you predicted 0 beds when 1 was required, dᵢ=1, so sᵢ = 1−1.0 = 0 (total failure).

### 2. The weight: wᵢ

Each item category has a weight reflecting how important it is:

- **wᵢ = 1.0** for essential items (bed, end table, dresser)
- **wᵢ = 0.5** for optional items (TV)

This means forgetting an essential item drags the score down twice as hard as forgetting an optional one.

### 3. The weighted average: Σ wᵢsᵢ / Σ wᵢ

**(Σ wᵢsᵢ + Σ wᵍsᵍ) / (Σ wᵢ + Σ wᵍ)**

- The top (numerator) sums up each item's score multiplied by its importance weight. A perfect essential item contributes 1.0×1.0 = 1.0. A perfect optional item contributes 0.5×1.0 = 0.5.
- The bottom (denominator) sums up all the weights. For a master bedroom: 1.0 + 1.0 + 1.0 + 0.5 = 3.5.
- This is just a **weighted average** — it asks "across all items, weighted by importance, what fraction of the inventory did we get right?"

The **g subscript** (wᵍ, sᵍ) is for **requirement groups** — cases where the room spec allows alternatives. A full bath needs "bathing" which can be satisfied by a tub OR a shower. A kitchen needs "cooking" which can be a range OR an oven+cooktop. The group score sᵍ is 1 if any alternative is present, 0 otherwise. These groups get folded into the same weighted average alongside the individual items.

### 4. The extra penalty: p_extra

**− p_extra**

Some models hallucinate items that don't belong in the room spec at all — a toilet in a bedroom, for instance. These "unsupported extras" get a separate penalty that is subtracted from the weighted average.

In the current benchmark: `p_extra = min(1, 0.25 × n_extra / 4)` where n_extra is the number of unsupported items. It's deliberately capped and mild — a few weird extras won't tank the score, but many will.

### 5. The × 100

**Coverage = 100 × (...)**

Rescales from a 0–1 fraction to a 0–100 score for readability. A perfect room is 100.0.

### Concrete example: master bedroom

The spec says: bed 1..1 (w=1.0), end table 2..2 (w=1.0), dresser 1..1 (w=1.0), TV 0..1 (w=0.5).

**Case A — everything present** (1 bed, 2 end tables, 1 dresser, 1 TV):
- Every sᵢ = 1.0
- Numerator = 1.0×1 + 1.0×1 + 1.0×1 + 0.5×1 = 3.5
- Denominator = 3.5
- Coverage = 100 × (3.5/3.5 − 0) = **100.0**

**Case B — dresser and TV removed** (1 bed, 2 end tables, 0 dresser, 0 TV):
- bed: s=1, end table: s=1, dresser: d=1→s=0, TV: pred=0 but min=0 so d=0→s=1
- Numerator = 1.0×1 + 1.0×1 + 1.0×0 + 0.5×1 = 2.5
- Coverage = 100 × (2.5/3.5 − 0) = **71.4**

The dresser (essential, w=1.0) being missing is what tanks the score. The TV at 0 is fine because the spec allows 0..1.

## Shared code

`explainer_common/` (one level up) provides `configure_rc()`, `rounded_box()`, and `save()` — used by all three explainer scripts to avoid duplication.

## Relationship to other explainers

- **`navigability_explainer/`** — sibling directory, covers the navigability metric.
- **`overlap_clearance_explainer/`** — sibling directory, covers the OC metric.
- Each explainer generates its own assets independently. The only shared code is `explainer_common/`.

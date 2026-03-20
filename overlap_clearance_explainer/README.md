# Overlap & Clearance Metric Explainer

Standalone blogpost-style page explaining the **Overlap & Clearance (OC)** benchmark metric from the ECCV 2026 SBM paper.

## Viewing

Open `index.html` in a browser. No build step required — all assets are pre-generated static files.

## Asset generation

All figures are produced by `generate_oc_assets.py`. Run:

```bash
python generate_oc_assets.py
```

### Dependencies

- Python 3.10+
- `matplotlib`, `numpy` (standard scientific stack)

### What gets generated

| Asset | Generator | External data needed? |
|---|---|---|
| `fig_01_oc_components.svg` | `render_components()` | No |
| `fig_02_overlap_eligibility.svg` | `render_eligibility()` | No |
| `fig_03_oc_geometry.svg` | `render_synthetic_oc()` | No |
| `fig_04_oc_case_studies.png` | `render_case_studies()` | Yes |
| `fig_05_oc_leaderboard.svg` | `render_leaderboard()` | Yes (JSON only) |

### External data

Figures 01–03 use synthetic geometry — no external files needed.

Figure 04 (the real benchmark case studies) requires:

1. The **benchmark repo** at `ai-services/tools/benchmarks/lbm-ddep` — specifically the OC metric code, heldout query inputs, and DDEP prediction outputs.
2. Three specific rooms: `XvLpePHYAIibaGeI` (clean), `QmZjhBBDVWwYWEoU` (door intrusion + overlap), and `y1R3RQ0zbpx3SYgN` (wall-bounds violation).

Figure 05 (the leaderboard) only needs the exported JSON at `results/paper_exports/camera_ready_metrics_export.json`.

If the benchmark imports are unavailable, the script skips the case-study figure gracefully and still generates figures 01–03 and 05.

## The OC formula explained

**OC % = 100 × (0.5 × EOF + 0.2 × GOA + 0.3 × DCI + 0.0 × WBV)**

### 1. Entity Overlap Fraction (EOF)

For each entity, EOF measures how much of its footprint is occupied by eligible neighbors.

- Only physically implausible overlaps count: prop-prop and same-category casework overlaps are eligible. Prop-casework overlaps are ignored (they often represent intentional arrangements).
- EOF is averaged across all entities in the room.

### 2. Global Overlap Area (GOA)

GOA measures the total multiply-occupied floor area as a fraction of the room area.

- Where EOF is per-entity, GOA is a single global number that captures total collision area.

### 3. Door Clearance Intrusion (DCI)

DCI measures how much of the door swing/clearance zone is blocked by furniture.

- The clearance zone is the region immediately inside each door opening.
- DCI is the fraction of that zone that is intruded upon by entity footprints.

### 4. Wall-Bounds Violation (WBV)

WBV measures how much furniture extends outside the room polygon.

- **WBV carries weight zero in the composite** because wall-boundary violations are strongly correlated with EOF — entities protruding outside the room also register as overlapping with the room boundary.
- WBV is still computed and reported as a separate diagnostic.

### 5. The composite weights

The non-zero weights reflect benchmark priorities:

- **0.5 × EOF** — Object-object collisions matter most.
- **0.3 × DCI** — Doorway blockage matters next.
- **0.2 × GOA** — Global overlap area is a smaller correction on top of per-entity overlap.

### Concrete example

A room with EOF = 0.02, GOA = 0.00, DCI = 0.04:

OC % = 100 × (0.5 × 0.02 + 0.2 × 0.00 + 0.3 × 0.04) = 2.2

Lower OC is better — it means less collision geometry and clearer door zones.

## Shared code

`explainer_common/` (one level up) provides `configure_rc()`, `rounded_box()`, and `save()` — used by all three explainer scripts to avoid duplication.

## Relationship to other explainers

- **`coverage_explainer/`** — sibling directory, covers the coverage metric.
- **`navigability_explainer/`** — sibling directory, covers the navigability metric.
- Each explainer generates its own assets independently. The only shared code is `explainer_common/`.

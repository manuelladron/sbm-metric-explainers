# Navigability Metric Explainer

Standalone blogpost-style page explaining the **Navigability** benchmark metric from the ECCV 2026 SBM paper.

## Viewing

Open `index.html` in a browser. No build step required — all assets are pre-generated static files.

## Asset generation

All figures are produced by `generate_navigability_assets.py`. Run:

```bash
python generate_navigability_assets.py
```

### Dependencies

- Python 3.10+
- `matplotlib`, `numpy` (standard scientific stack)

### What gets generated

| Asset | Generator | External data needed? |
|---|---|---|
| `fig_00_legend_strip.svg` | `build_legend_strip()` | No |
| `fig_01_navigability_pipeline.svg` | `build_pipeline_figure()` | No |
| `fig_02_door_portal.svg` | `build_door_portal_figure()` | No |
| `fig_03_target_construction.svg` | `build_targets_figure()` | No |
| `fig_04_clearance_and_walkable_space.svg` | `build_clearance_figure()` | No |
| `fig_05_reachability_vs_detour.svg` | `build_reachability_figure()` | No |
| `fig_06_score_decomposition.svg` | `build_score_figure()` | No |
| `fig_07_controlled_benchmark.svg` | `build_benchmark_figure()` | No |
| `fig_08_real_benchmark_case.png` | `build_benchmark_case_figure()` | Yes |

### External data

Figures 01–07 use synthetic room geometry and hardcoded benchmark numbers — no external files needed. They do import geometry helpers from the benchmark repo at `ai-services/tools/benchmarks/lbm-ddep` (specifically `metrics.navigation_metric.geometry`).

Figure 08 (the real benchmark case study) requires:

1. The **benchmark repo** at `ai-services/tools/benchmarks/lbm-ddep` — the heldout query inputs, DDEP navigability plot output, and per-room metric JSON.
2. Specifically: room `XvLpePHYAIibaGeI` from the `MASTER_BED` heldout set.

If the benchmark paths are unavailable, the script skips figure 08 gracefully and still generates figures 00–07.

## The Navigability formula explained

**Nav = 100 × (SR − λ × DF)**

### 1. Door portals and essential targets

For every room, the benchmark identifies **door portals** (entry points offset inward from each door opening) and **essential targets** (points near required furniture). Every door-target pair is then evaluated.

- **Beds** contribute two targets (left and right side access): T = { C ± (wₑ / 2 + r + ε) x }
- **Other essential items** (sinks, toilets, ranges, dressers) contribute one front-access target: T = { A + (dₑ + r + ε) y }

### 2. Success Rate: SR

**SR(r) = (1 / |D| |T|) × Σ δ(d, t)**

- For each door-target pair, A* pathfinding runs on a rasterized walkable grid.
- **δ(d, t) = 1** if a collision-free path exists, **0** otherwise.
- SR is the fraction of all pairs that are reachable.

### 3. Detour Factor: DF

**ρ(d, t) = L(d, t) / max(‖t − d‖, ε)**

- **ρ** is the ratio of actual path length to straight-line distance. ρ = 1 means a perfectly direct path.
- **f(ρ) = min(1, (ρ − 1) / 2)** — the detour penalty per pair, capped at 1.
- **DF(r)** = average of f(ρ) over reachable pairs only.

### 4. The composite

**Nav = 100 × (SR − 0.35 × DF)**

- **λ = 0.35** is the fixed tradeoff weight.
- A room where every target is reachable via direct paths scores 100.
- Negative scores are possible: if SR is very low and the few surviving paths are highly circuitous, the detour penalty can overpower SR.

### 5. Walkable grid construction

- The room polygon is rasterized at ~320 cells on the long side.
- Each furniture footprint is **inflated** by a human clearance radius (r = corridor_width / 2 = 1 ft by default).
- Walkable cells = room mask minus inflated obstacles.
- The door portal is snapped to the nearest walkable cell if the candidate position falls on a blocked cell.

## Shared code

`explainer_common/` (one level up) provides `configure_rc()`, `rounded_box()`, and `save()` — used by all three explainer scripts to avoid duplication.

## Relationship to other explainers

- **`coverage_explainer/`** — sibling directory, covers the coverage metric.
- **`overlap_clearance_explainer/`** — sibling directory, covers the OC metric.
- Each explainer generates its own assets independently. The only shared code is `explainer_common/`.

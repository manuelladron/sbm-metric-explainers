from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

# Shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainer_common import configure_rc, rounded_box, save as _save_fig


BENCHMARK_ROOT = Path(
    "/Users/manuelrodriguez/Documents/higharc/gits/ai-services/tools/benchmarks/lbm-ddep"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
BENCHMARK_EXPORT = BENCHMARK_ROOT / "results" / "paper_exports" / "camera_ready_metrics_export.json"
HELDOUT_INPUTS = BENCHMARK_ROOT / "heldout_query_set_eccv" / "layout_inputs" / "texts"
DDEP_HELDOUT = BENCHMARK_ROOT / "results" / "eccv_heldout" / "ddep_v1_predictions_processed"
CONFIG_PATH = BENCHMARK_ROOT / "configs" / "guided_inventory_metrics.yaml"
VOCAB_DIR = (
    BENCHMARK_ROOT.parent.parent.parent
    / "ml_models"
    / "LBM"
    / "LBM_v1"
    / "lbm_data"
    / "models"
)

# Benchmark metric imports — only needed for case-study figures (masterbed, laundry).
# Guarded so that the simpler figures can still be regenerated without the benchmark repo.
_HAS_BENCHMARK = False
try:
    sys.path.insert(0, str(BENCHMARK_ROOT))
    from metrics.coverage_metric.coverage import (  # noqa: E402
        CoverageCalculator,
        EntityNameNormalizer,
        load_inventory_spec,
    )
    from metrics.navigation_metric.geometry import (  # noqa: E402
        edge_frames_feet,
        oriented_rect,
        room_poly_feet,
    )

    room_specs, defaults, aliases = load_inventory_spec(CONFIG_PATH)
    normalizer = EntityNameNormalizer(VOCAB_DIR, aliases)
    coverage_calculator = CoverageCalculator(room_specs, defaults, normalizer)
    _HAS_BENCHMARK = True
except Exception:
    pass


COLORS = {
    "paper": "#F6F1E8",
    "panel": "#FFFDFC",
    "panel_soft": "#FBF7EF",
    "ink": "#131313",
    "muted": "#66615B",
    "line": "#D8D0C4",
    "coverage": "#B86A3C",
    "coverage_soft": "#F2DEC6",
    "accent": "#1D6F72",
    "nav": "#1D6F72",
    "nav_soft": "#DDECEC",
    "gold": "#C9A14F",
    "gold_soft": "#F3E8C9",
    "wall": "#282828",
    "room_fill": "#F7F2E7",
    "door": "#C06A38",
    "window": "#5AA4D8",
    "entity": "#C7D8DE",
    "entity_optional": "#EEE4CF",
    "entity_edge": "#60737C",
}

LABELS = {
    "bed": "bed",
    "endtable": "end table",
    "dresser": "dresser",
    "tv": "TV",
    "washer": "washer",
    "dryer": "dryer",
    "closet_shelf": "shelf",
    "sink": "sink",
    "range": "range",
    "oven": "oven",
    "rangetop": "cooktop",
    "shower": "shower",
    "tub": "tub",
    "base_cabinet": "cabinet",
    "tv_cabinet": "TV cabinet",
}


# ---------------------------------------------------------------------------
# Dataclass for room case studies
# ---------------------------------------------------------------------------

@dataclass
class RoomBundle:
    room_id: str
    room_type: str
    input_payload: dict
    output_payload: dict
    coverage: dict
    navigability: dict | None
    normalized_entities: list[tuple[str, str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    _save_fig(fig, OUTPUT_DIR / name)


def pretty(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Benchmark-dependent helpers (room rendering, entity drawing, etc.)
# ---------------------------------------------------------------------------

def essential_categories(room_type: str) -> set[str]:
    spec = room_specs[room_type]
    categories = {
        item.category
        for item in spec.props + spec.casework
        if item.essential
    }
    categories.update(spec.requirement_groups.keys())
    return categories


def load_bundle(room_id: str) -> RoomBundle:
    input_payload = json.loads((HELDOUT_INPUTS / f"{room_id}.json").read_text())
    output_payload = json.loads(
        (DDEP_HELDOUT / "unguided" / "text" / f"{room_id}.json").read_text()
    )
    coverage_payload = json.loads((DDEP_HELDOUT / "metrics_cov" / "coverage.json").read_text())
    nav_payload = json.loads((DDEP_HELDOUT / "metrics_nav" / "navigability.json").read_text())
    room_type = output_payload["room_type"]
    normalized_entities = []
    for entity in output_payload.get("entities", []):
        raw = entity.get("category", "")
        normalized_entities.append((raw, normalizer.normalize(raw, room_type)))
    return RoomBundle(
        room_id=room_id,
        room_type=room_type,
        input_payload=input_payload,
        output_payload=output_payload,
        coverage=coverage_payload["rooms"][room_id],
        navigability=nav_payload["rooms"].get(room_id),
        normalized_entities=normalized_entities,
    )


def room_limits(poly: np.ndarray, pad: float = 0.65) -> tuple[tuple[float, float], tuple[float, float]]:
    min_x, min_y = poly.min(axis=0)
    max_x, max_y = poly.max(axis=0)
    return (min_x - pad, max_x + pad), (min_y - pad, max_y + pad)


def draw_room_outline(ax: plt.Axes, room_payload: dict) -> tuple[np.ndarray, dict]:
    room_dict = room_payload["room"]
    poly = room_poly_feet(room_dict)
    frames = edge_frames_feet(room_dict)
    ax.add_patch(
        Polygon(
            poly,
            closed=True,
            facecolor=COLORS["room_fill"],
            edgecolor=COLORS["wall"],
            linewidth=2.1,
            zorder=1,
        )
    )
    for opening in room_dict.get("openings", []):
        edge = frames.get(opening["edge_id"])
        if edge is None:
            continue
        start, end, _, _, _ = edge
        seg_a = start + opening["t_start"] * (end - start)
        seg_b = start + opening["t_end"] * (end - start)
        if opening.get("type") == "DOOR":
            color = COLORS["door"]
            width = 4.0
        else:
            color = COLORS["window"]
            width = 3.0
        ax.plot(
            [seg_a[0], seg_b[0]],
            [seg_a[1], seg_b[1]],
            color=color,
            linewidth=width,
            solid_capstyle="round",
            zorder=6,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    (xmin, xmax), (ymin, ymax) = room_limits(poly)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return poly, frames


def entity_face(normalized: str, room_type: str) -> str:
    if normalized in essential_categories(room_type):
        return COLORS["entity"]
    return COLORS["entity_optional"]


def draw_entities(ax: plt.Axes, bundle: RoomBundle, with_labels: bool = True) -> None:
    _, frames = draw_room_outline(ax, bundle.input_payload)
    for entity in bundle.output_payload.get("entities", []):
        if entity.get("edge_id") not in frames:
            continue
        rect = oriented_rect(entity, frames)
        raw = entity.get("category", "")
        normalized = normalizer.normalize(raw, bundle.room_type)
        ax.add_patch(
            Polygon(
                rect,
                closed=True,
                facecolor=entity_face(normalized, bundle.room_type),
                edgecolor=COLORS["entity_edge"],
                linewidth=1.2,
                zorder=3,
            )
        )
        if with_labels:
            center = rect.mean(axis=0)
            ax.text(
                center[0],
                center[1],
                pretty(normalized),
                ha="center",
                va="center",
                fontsize=8.1,
                color=COLORS["ink"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": COLORS["panel"],
                    "edgecolor": COLORS["line"],
                    "linewidth": 0.7,
                },
                zorder=7,
            )


def add_card_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=13, weight="bold")
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"])


def add_plan_card_text(ax: plt.Axes, lines: list[tuple[str, str]]) -> None:
    ax.axis("off")
    rounded_box(ax, 0.0, 0.02, 1.0, 0.96, COLORS["panel"], COLORS["line"], radius=0.05)
    y = 0.84
    val_x = 0.34
    for label, value in lines:
        ax.text(0.06, y, label, fontsize=10.5, color=COLORS["muted"])
        if len(value) > 30 and ", " in value:
            parts = value.split(", ")
            mid = (len(parts) + 1) // 2
            ax.text(val_x, y, ", ".join(parts[:mid]) + ",", fontsize=11.5, weight="bold", color=COLORS["ink"])
            y -= 0.12
            ax.text(val_x, y, ", ".join(parts[mid:]), fontsize=11.5, weight="bold", color=COLORS["ink"])
        else:
            ax.text(val_x, y, value, fontsize=11.5, weight="bold", color=COLORS["ink"])
        y -= 0.16


def score_lines(bundle: RoomBundle) -> list[tuple[str, str]]:
    coverage = bundle.coverage["coverage_score"]
    normalized_counter: dict[str, int] = {}
    for _, canonical in bundle.normalized_entities:
        normalized_counter[canonical] = normalized_counter.get(canonical, 0) + 1
    summary_parts = [f"{pretty(k)} x{v}" for k, v in normalized_counter.items()]
    weighted = bundle.coverage["total_weighted_score"]
    total_weight = bundle.coverage["total_weight"]
    lines = [
        ("Coverage", f"{coverage:.1f}"),
        ("Formula", f"{weighted:.1f} / {total_weight:.1f}"),
        ("Counts", ", ".join(summary_parts) if summary_parts else "none"),
    ]
    if bundle.coverage["missing_items"]:
        miss = ", ".join(pretty(item["category"]) for item in bundle.coverage["missing_items"])
        lines.append(("Missing", miss))
    return lines


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------

def _draw_breakdown(ax: plt.Axes) -> None:
    """Draw the three-card coverage breakdown diagram on *ax*."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card_w = 0.28
    card_h = 0.80
    card_y = 0.08
    gap = 0.04
    total_w = 3 * card_w + 2 * gap
    margin = (1 - total_w) / 2
    cx = [margin + i * (card_w + gap) for i in range(3)]
    row_w = card_w - 0.04
    title_y = card_y + card_h - 0.08

    for x in cx:
        rounded_box(ax, x, card_y, card_w, card_h, COLORS["panel"], COLORS["line"])

    # Card 1: Define the spec
    ax.text(cx[0] + 0.02, title_y, "1. Define the spec", fontsize=15.5, weight="bold")
    rows = [
        ("bed", "1..1", "essential", "w=1.0"),
        ("end table", "2..2", "essential", "w=1.0"),
        ("dresser", "1..1", "essential", "w=1.0"),
        ("TV", "0..1", "optional", "w=0.5"),
    ]
    y = title_y - 0.12
    for name, rng, kind, weight in rows:
        rounded_box(ax, cx[0] + 0.02, y - 0.045, row_w, 0.09, COLORS["panel"], COLORS["line"])
        ax.text(cx[0] + 0.04, y + 0.01, name, fontsize=12, weight="bold")
        ax.text(cx[0] + 0.15, y + 0.01, rng, fontsize=11, color=COLORS["muted"])
        ax.text(cx[0] + 0.04, y - 0.025, kind, fontsize=9.8, color=COLORS["muted"])
        ax.text(cx[0] + 0.20, y - 0.025, weight, fontsize=10, color=COLORS["coverage"])
        y -= 0.13

    # Card 2: Normalize names
    ax.text(cx[1] + 0.02, title_y, "2. Normalize names", fontsize=15.5, weight="bold")
    box_h = 0.22
    box_y = title_y - 0.12 - box_h
    rounded_box(ax, cx[1] + 0.02, box_y, row_w, box_h, COLORS["coverage_soft"], COLORS["coverage"])
    name_x = cx[1] + 0.04
    arrow_x = cx[1] + 0.155
    my = box_y + box_h - 0.055
    for orig, norm in [("King", "bed"), ("Nightstand", "end table"), ("Cabinet", "dresser")]:
        ax.text(name_x, my, orig, fontsize=12, weight="bold")
        ax.text(arrow_x, my, f"→ {norm}", fontsize=11.5)
        my -= 0.065
    ax.text(
        cx[1] + 0.02,
        box_y - 0.10,
        "Inside MASTER_BED, cabinet,\nbase_cabinet, and tv_cabinet\ncount as the storage requirement.",
        fontsize=10.5,
        color=COLORS["muted"],
        linespacing=1.45,
    )

    # Card 3: Penalize errors
    ax.text(cx[2] + 0.02, title_y, "3. Penalize errors", fontsize=15.5, weight="bold")
    fx = cx[2] + 0.025
    fy = title_y - 0.14
    for label, formula in [
        ("missing fraction", "d = max(0, min - pred) / max(1, min)"),
        ("overfill fraction", "o = max(0, pred - max) / max(1, max)"),
        ("item score", "s = 1 - clip(1.0·d + 0.5·o, 0, 1)"),
    ]:
        ax.text(fx, fy, label, fontsize=10.5, color=COLORS["muted"])
        ax.text(fx, fy - 0.06, formula, fontsize=11.8)
        fy -= 0.18


def render_breakdown() -> None:
    for ext in ("svg", "png"):
        fig, ax = plt.subplots(figsize=(16, 7))
        _draw_breakdown(ax)
        save(fig, f"fig_01_coverage_breakdown.{ext}")


def render_item_score_curves() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    counts = np.arange(0, 5)

    def item_score(pred: int, min_i: int, max_i: int, alpha: float = 1.0, beta: float = 0.5) -> float:
        d_i = max(0, min_i - pred) / max(1, min_i)
        o_i = max(0, pred - max_i) / max(1, max_i)
        return max(0.0, min(1.0, 1.0 - (alpha * d_i + beta * o_i)))

    required = [item_score(v, 1, 1) for v in counts]
    optional = [item_score(v, 0, 1) for v in counts]

    ax.plot(counts, required, color=COLORS["coverage"], marker="o", linewidth=2.4, label="required item: 1..1")
    ax.plot(counts, optional, color=COLORS["nav"], marker="o", linewidth=2.4, label="optional item: 0..1")
    ax.fill_between(counts, required, color=COLORS["coverage_soft"], alpha=0.5)
    ax.fill_between(counts, optional, color=COLORS["nav_soft"], alpha=0.45)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("predicted count")
    ax.set_ylabel("item score")
    ax.grid(color=COLORS["line"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="lower center", ncol=2)
    ax.set_title("Coverage is harsh on missing essentials and gentler on overfill", fontsize=14, weight="bold", pad=14)
    save(fig, "fig_02_item_score_curves.svg")


def render_requirement_groups() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top card: full bath
    rounded_box(ax, 0.04, 0.53, 0.92, 0.43, COLORS["panel"], COLORS["line"], radius=0.03)
    ax.text(0.08, 0.89, "Full bath", fontsize=20, weight="bold")
    ax.text(
        0.08, 0.83,
        "The room must support bathing, but the benchmark allows alternatives.",
        fontsize=12, color=COLORS["muted"],
    )
    bw, bh = 0.28, 0.10
    by = 0.66
    rounded_box(ax, 0.10, by, bw, bh, COLORS["nav_soft"], COLORS["nav"], radius=0.02)
    rounded_box(ax, 0.54, by, bw, bh, COLORS["coverage_soft"], COLORS["coverage"], radius=0.02)
    ax.text(0.10 + bw / 2, by + bh / 2, "tub", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.54 + bw / 2, by + bh / 2, "shower", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.45, by + bh / 2, "OR", ha="center", va="center", fontsize=14, color=COLORS["muted"])
    ax.text(0.08, 0.58, r"$s_g = 1$ if any alternative is satisfied; $0$ otherwise.", fontsize=13)

    # Bottom card: kitchen cooking
    rounded_box(ax, 0.04, 0.04, 0.92, 0.43, COLORS["panel"], COLORS["line"], radius=0.03)
    ax.text(0.08, 0.40, "Kitchen cooking", fontsize=20, weight="bold")
    ax.text(
        0.08, 0.34,
        "A single range works, but oven + cooktop works too.",
        fontsize=12, color=COLORS["muted"],
    )
    rbh_tall = 0.14
    rby = 0.14
    rounded_box(ax, 0.10, rby, bw, rbh_tall, COLORS["nav_soft"], COLORS["nav"], radius=0.02)
    ax.text(0.10 + bw / 2, rby + rbh_tall / 2, "range", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.45, rby + rbh_tall / 2, "OR", ha="center", va="center", fontsize=14, color=COLORS["muted"])
    rbh_half = 0.06
    rbh_gap = 0.02
    rounded_box(ax, 0.54, rby + rbh_half + rbh_gap, bw, rbh_half, COLORS["coverage_soft"], COLORS["coverage"], radius=0.02)
    rounded_box(ax, 0.54, rby, bw, rbh_half, COLORS["gold_soft"], COLORS["gold"], radius=0.02)
    ax.text(0.54 + bw / 2, rby + rbh_half + rbh_gap + rbh_half / 2, "oven", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.54 + bw / 2, rby + rbh_half / 2, "cooktop", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.08, 0.08, "Coverage measures room function, not just category counts.", fontsize=13)

    save(fig, "fig_03_requirement_groups.svg")


def render_masterbed_case() -> None:
    import copy

    good = load_bundle("XvLpePHYAIibaGeI")

    partial = copy.deepcopy(good)
    keep_norm = {"bed", "endtable"}
    partial.output_payload["entities"] = [
        e for e in partial.output_payload["entities"]
        if normalizer.normalize(e.get("category", ""), partial.room_type) in keep_norm
    ]
    partial.normalized_entities = [
        (raw, canon) for raw, canon in partial.normalized_entities
        if canon in keep_norm
    ]
    partial.coverage = {
        "coverage_score": 71.4,
        "total_weighted_score": 2.5,
        "total_weight": 3.5,
        "item_scores": {},
        "missing_items": [{"category": "dresser"}],
    }
    partial.navigability = None

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.15], hspace=0.18, wspace=0.16)
    plan_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    text_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    bundles = [good, partial]
    titles = [
        ("Case A: full inventory present", "MASTER_BED · Coverage 100.0"),
        ("Case B: dresser and TV removed", "MASTER_BED · Coverage 71.4"),
    ]
    for ax, bundle, title in zip(plan_axes, bundles, titles):
        draw_entities(ax, bundle, with_labels=True)
        add_card_title(ax, *title)
    for ax, bundle in zip(text_axes, bundles):
        add_plan_card_text(ax, score_lines(bundle))
    fig.suptitle("Coverage case study: same room, different inventories", fontsize=22, weight="bold", x=0.5, y=0.99)
    save(fig, "fig_04_masterbed_coverage_case.png")


def render_laundry_case() -> None:
    bundle = load_bundle("om5k9YFcVPJWUY0Q")
    fig = plt.figure(figsize=(12.6, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.12)
    ax_room = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    draw_entities(ax_room, bundle, with_labels=True)
    add_card_title(ax_room, "A room can be easy to move through and still fail coverage", "om5k9YFcVPJWUY0Q · LAUNDRY")

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    rounded_box(ax_text, 0.02, 0.06, 0.96, 0.88, COLORS["panel"], COLORS["line"], radius=0.04)
    ax_text.text(0.07, 0.84, "What the benchmark sees", fontsize=19, weight="bold")
    ax_text.text(0.07, 0.70, "Requested inventory", fontsize=11, color=COLORS["muted"])
    ax_text.text(0.07, 0.63, "washer + dryer", fontsize=13.2, weight="bold")
    ax_text.text(0.07, 0.57, "(optional shelf allowed)", fontsize=10.8, color=COLORS["muted"])
    ax_text.text(0.07, 0.48, "Predicted after normalization", fontsize=11, color=COLORS["muted"])
    ax_text.text(0.07, 0.41, "washer + shelf", fontsize=13.2, weight="bold")
    ax_text.text(0.52, 0.43, "Coverage", fontsize=11, color=COLORS["muted"])
    ax_text.text(0.52, 0.33, f"{bundle.coverage['coverage_score']:.1f}", fontsize=22, weight="bold", color=COLORS["coverage"])
    ax_text.text(0.75, 0.43, "Navigability", fontsize=11, color=COLORS["muted"])
    ax_text.text(0.75, 0.33, f"{bundle.navigability['NavScore']:.1f}", fontsize=22, weight="bold", color=COLORS["nav"])
    ax_text.text(
        0.07,
        0.18,
        "The washer target is reachable from the\ndoor, so navigation is excellent. Coverage\nstill drops because the required dryer\nnever appears.",
        fontsize=11.5,
        linespacing=1.45,
    )
    save(fig, "fig_05_laundry_complementarity_case.png")


def render_coverage_leaderboard() -> None:
    export = json.loads(BENCHMARK_EXPORT.read_text())
    rows = []
    for row in export["table1_rows"]:
        value = row.get("coverage_mean")
        if value is None:
            continue
        mode = ""
        if row["kind"] == "frontier":
            mode = " (LLM)" if row["section"] == "text" else " (VLM)"
        rows.append((f"{row['display']}{mode}", float(value), row["kind"]))

    rows.sort(key=lambda item: item[1], reverse=True)

    names = [name for name, _, _ in rows[:10]]
    values = [value for _, value, _ in rows[:10]]
    colors = [COLORS["coverage"] if kind == "ddep" else COLORS["coverage_soft"] for _, _, kind in rows[:10]]
    edges = [COLORS["coverage"] if kind != "ddep" else COLORS["ink"] for _, _, kind in rows[:10]]

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ypos = list(range(len(names)))
    ax.barh(ypos, values, color=colors, edgecolor=edges, linewidth=1.2)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xlabel("coverage mean")
    ax.set_title("Current benchmark means on the coverage metric", fontsize=14.5, weight="bold", pad=14)
    ax.grid(axis="x", color=COLORS["line"], linestyle="--", linewidth=0.8, alpha=0.8)
    for y, value in zip(ypos, values):
        ax.text(value + 1.0, y, f"{value:.1f}", va="center", fontsize=9.4)
    save(fig, "fig_06_coverage_leaderboard.svg")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_rc()

    # Always-available figures (no benchmark data needed)
    render_breakdown()
    render_item_score_curves()
    render_requirement_groups()
    render_coverage_leaderboard()

    # Case-study figures (require benchmark repo access)
    if _HAS_BENCHMARK:
        render_masterbed_case()
        render_laundry_case()
    else:
        print("skipped case-study figures (benchmark imports unavailable)")


if __name__ == "__main__":
    main()

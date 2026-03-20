from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# Shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainer_common import configure_rc, rounded_box, save as _save_fig


BENCHMARK_ROOT = Path(
    "/Users/manuelrodriguez/Documents/higharc/gits/ai-services/tools/benchmarks/lbm-ddep"
)
HELDOUT_INPUTS = BENCHMARK_ROOT / "heldout_query_set_cvpr" / "layout_inputs" / "texts"
DDEP_HELDOUT = BENCHMARK_ROOT / "results" / "eccv_heldout" / "ddep_v1_predictions_processed"
EXPORT_JSON = BENCHMARK_ROOT / "results" / "paper_exports" / "camera_ready_metrics_export.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"

import sys

sys.path.insert(0, str(BENCHMARK_ROOT))

from metrics.overlap_clearance_metric.oc_metric import (  # noqa: E402
    build_door_clearance_zone,
    build_entity_polygon,
    edge_frames_inches,
    is_eligible_pair,
    load_overlap_config,
    oriented_rect_inches,
    room_poly_inches,
)
from metrics.navigation_metric.nav_metric import load_dimensions_fallback  # noqa: E402


COLORS = {
    "paper": "#F6F1E8",
    "panel": "#FFFDFC",
    "panel_soft": "#FBF7EF",
    "ink": "#131313",
    "muted": "#66615B",
    "line": "#D8D0C4",
    "eof": "#B64F3F",
    "eof_soft": "#F2D6CF",
    "goa": "#B9892D",
    "goa_soft": "#F4E6BF",
    "dci": "#1D6F72",
    "dci_soft": "#DDECEC",
    "wbv": "#6A5ACD",
    "wbv_soft": "#E6E1F6",
    "room_fill": "#F7F2E7",
    "wall": "#282828",
    "door": "#C06A38",
    "entity": "#C7D8DE",
    "entity_edge": "#60737C",
}


@dataclass
class RoomBundle:
    room_id: str
    room_type: str
    room_json: dict
    pred_json: dict
    metrics: dict


DIMENSIONS = load_dimensions_fallback()
OVERLAP_CONFIG = load_overlap_config()
ALIASES = {}


def save(fig: plt.Figure, name: str) -> None:
    _save_fig(fig, OUTPUT_DIR / name)


def load_bundle(room_id: str) -> RoomBundle:
    room_json = json.loads((HELDOUT_INPUTS / f"{room_id}.json").read_text())
    pred_json = json.loads((DDEP_HELDOUT / "unguided" / "text" / f"{room_id}.json").read_text())
    metrics = json.loads((DDEP_HELDOUT / "metrics_oc" / "overlap_clearance.json").read_text())["rooms"][room_id]
    return RoomBundle(room_id, pred_json["room_type"], room_json, pred_json, metrics)


def inches_to_feet_coords(coords):
    return [(x / 12.0, y / 12.0) for x, y in coords]


def shapely_to_patch(ax: plt.Axes, geom, face: str, edge: str, alpha: float = 0.65, lw: float = 1.0, zorder: int = 5) -> None:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        coords = inches_to_feet_coords(list(geom.exterior.coords))
        ax.add_patch(Polygon(coords, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            shapely_to_patch(ax, part, face, edge, alpha=alpha, lw=lw, zorder=zorder)


def room_limits(poly_inches: np.ndarray, pad_ft: float = 0.7):
    pts = poly_inches / 12.0
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    return (min_x - pad_ft, max_x + pad_ft), (min_y - pad_ft, max_y + pad_ft)


def draw_room_base(ax: plt.Axes, bundle: RoomBundle):
    room = bundle.room_json["room"]
    poly_in = room_poly_inches(room)
    frames = edge_frames_inches(room)
    ax.add_patch(
        Polygon(
            poly_in / 12.0,
            closed=True,
            facecolor=COLORS["room_fill"],
            edgecolor=COLORS["wall"],
            linewidth=2.0,
            zorder=1,
        )
    )
    for opening in room.get("openings", []):
        edge = frames.get(opening["edge_id"])
        if edge is None:
            continue
        start, end, _, _, _ = edge
        seg_a = start + opening["t_start"] * (end - start)
        seg_b = start + opening["t_end"] * (end - start)
        color = COLORS["door"] if opening.get("type") == "DOOR" else "#5AA4D8"
        width = 4.0 if opening.get("type") == "DOOR" else 3.0
        ax.plot([seg_a[0] / 12.0, seg_b[0] / 12.0], [seg_a[1] / 12.0, seg_b[1] / 12.0], color=color, linewidth=width, solid_capstyle="round", zorder=6)

    entity_polys = []
    for ent in bundle.pred_json.get("entities", []):
        if ent.get("anchor_type") != "wall_anchored":
            continue
        poly = build_entity_polygon(ent, frames, DIMENSIONS)
        if poly is None or poly.is_empty:
            continue
        shapely_to_patch(ax, poly, COLORS["entity"], COLORS["entity_edge"], alpha=0.9, lw=1.2, zorder=3)
        entity_polys.append(poly)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    (xmin, xmax), (ymin, ymax) = room_limits(poly_in)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return frames, entity_polys, poly_in


def highlight_dci(ax: plt.Axes, bundle: RoomBundle, frames, entity_polys) -> None:
    room = bundle.room_json["room"]
    doors = [d for d in room.get("openings", []) if d.get("type") == "DOOR"]
    if not doors:
        return
    union = unary_union(entity_polys) if entity_polys else None
    for door in doors:
        zone = build_door_clearance_zone(door, frames, 36.0)
        shapely_to_patch(ax, zone, COLORS["dci_soft"], COLORS["dci"], alpha=0.55, lw=1.2, zorder=2)
        if union is not None:
            try:
                intrusion = zone.intersection(union)
            except Exception:
                intrusion = None
            shapely_to_patch(ax, intrusion, COLORS["eof"], COLORS["eof"], alpha=0.65, lw=0.8, zorder=7)


def highlight_eof_and_wbv(ax: plt.Axes, bundle: RoomBundle, frames, entity_polys, poly_in: np.ndarray) -> None:
    intersections = []
    entities = bundle.pred_json.get("entities", [])
    for i, poly_i in enumerate(entity_polys):
        for j, poly_j in enumerate(entity_polys):
            if i >= j:
                continue
            if is_eligible_pair(entities[i], entities[j], ALIASES, OVERLAP_CONFIG.get("allowed_overlaps", [])):
                inter = poly_i.intersection(poly_j)
                if not inter.is_empty and inter.area > 1e-6:
                    intersections.append(inter)
    if intersections:
        overlap_union = unary_union(intersections)
        shapely_to_patch(ax, overlap_union, COLORS["eof_soft"], COLORS["eof"], alpha=0.8, lw=1.0, zorder=7)

    room_poly = ShapelyPolygon([tuple(p) for p in poly_in])
    for poly in entity_polys:
        outside = poly.difference(room_poly)
        shapely_to_patch(ax, outside, COLORS["wbv_soft"], COLORS["wbv"], alpha=0.85, lw=1.0, zorder=8)


def render_components() -> None:
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card_w = 0.215
    gap = 0.025
    total = 4 * card_w + 3 * gap
    margin = (1 - total) / 2
    cx = [margin + i * (card_w + gap) for i in range(4)]
    pad = 0.02  # text inset from card edge

    cards = [
        (COLORS["eof"], COLORS["eof_soft"], "EOF", "Entity overlap\nfraction",
         "How much of each entity\nfootprint is occupied by\neligible neighbors."),
        (COLORS["goa"], COLORS["goa_soft"], "GOA", "Global overlap\narea",
         "How much room area\nbecomes multiply occupied\nwhen all eligible overlaps\nare unioned."),
        (COLORS["dci"], COLORS["dci_soft"], "DCI", "Door clearance\nintrusion",
         "How much of the inward\n3 ft doorway zone is\nblocked by furniture."),
        (COLORS["wbv"], COLORS["wbv_soft"], "WBV", "Wall-bounds\nviolation",
         "How much of each entity\nfootprint sticks outside\nthe room boundary.\nDiagnostic only."),
    ]
    for x, (edge, face, title, subtitle, copy) in zip(cx, cards):
        rounded_box(ax, x, 0.10, card_w, 0.82, COLORS["panel"], COLORS["line"])
        rounded_box(ax, x + pad, 0.76, 0.10, 0.10, face, edge, radius=0.025)
        ax.text(x + pad + 0.05, 0.81, title, ha="center", va="center", fontsize=16, weight="bold", color=edge)
        ax.text(x + pad, 0.56, subtitle, fontsize=13.5, weight="bold", linespacing=1.3)
        ax.text(x + pad, 0.18, copy, fontsize=11, color=COLORS["muted"], linespacing=1.5)

    save(fig, "fig_01_oc_components.svg")


def render_eligibility() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card_w = 0.45
    gap = 0.04
    total = 2 * card_w + gap
    margin = (1 - total) / 2
    lx = margin
    rx = margin + card_w + gap
    pad = 0.03

    rounded_box(ax, lx, 0.08, card_w, 0.84, COLORS["panel"], COLORS["line"])
    rounded_box(ax, rx, 0.08, card_w, 0.84, COLORS["panel"], COLORS["line"])

    # Left card: What counts
    ax.text(lx + pad, 0.82, "What counts as overlap", fontsize=17, weight="bold")

    ax.text(lx + pad, 0.70, "PROP ↔ PROP", fontsize=14.5, weight="bold", color=COLORS["eof"])
    ax.text(lx + pad, 0.63, "Always eligible.", fontsize=12.5)

    ax.text(lx + pad, 0.51, "CASEWORK ↔ CASEWORK", fontsize=14.5, weight="bold", color=COLORS["goa"])
    ax.text(lx + pad, 0.41, "Only when the canonical\ncategories match.", fontsize=12.5, color=COLORS["muted"], linespacing=1.4)

    ax.text(lx + pad, 0.26, "Example: base cabinet overlapping\nanother base cabinet counts.", fontsize=12, linespacing=1.4)

    # Right card: What does not count
    ax.text(rx + pad, 0.82, "What does not count", fontsize=17, weight="bold")

    ax.text(rx + pad, 0.70, "PROP ↔ CASEWORK", fontsize=14.5, weight="bold", color=COLORS["dci"])
    ax.text(rx + pad, 0.63, "Ignored.", fontsize=12.5)

    ax.text(rx + pad, 0.51, "Unless the pair is explicitly\ndisallowed elsewhere.", fontsize=12.5, color=COLORS["muted"], linespacing=1.4)

    ax.text(rx + pad, 0.30,
            "This is deliberate: a lamp on a\nnightstand should not be treated\nlike two sofas occupying the\nsame floor area.",
            fontsize=12, linespacing=1.45)

    save(fig, "fig_02_overlap_eligibility.svg")


def render_synthetic_oc() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card_w = 0.45
    gap = 0.04
    total = 2 * card_w + gap
    margin = (1 - total) / 2
    lx = margin
    rx = margin + card_w + gap
    pad = 0.03

    rounded_box(ax, lx, 0.08, card_w, 0.84, COLORS["panel"], COLORS["line"])
    rounded_box(ax, rx, 0.08, card_w, 0.84, COLORS["panel"], COLORS["line"])

    # Left card: EOF / GOA overlap geometry
    ax.text(lx + pad, 0.80, "EOF and GOA both come\nfrom overlap geometry", fontsize=16, weight="bold", linespacing=1.25)

    # Two overlapping entity boxes with highlighted intersection (compact)
    bx = lx + 0.08
    ax.add_patch(FancyBboxPatch((bx, 0.46), 0.14, 0.22, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=COLORS["entity"], edgecolor=COLORS["entity_edge"], linewidth=1.2))
    ax.add_patch(FancyBboxPatch((bx + 0.09, 0.50), 0.14, 0.22, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=COLORS["entity"], edgecolor=COLORS["entity_edge"], linewidth=1.2))
    ax.add_patch(FancyBboxPatch((bx + 0.09, 0.50), 0.05, 0.18, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=COLORS["eof_soft"], edgecolor=COLORS["eof"], linewidth=1.0))

    ax.text(lx + pad, 0.26,
            "EOF asks how much of each\nentity is overlapped. GOA asks\nhow much total room area\nbecomes multiply occupied.",
            fontsize=13, color=COLORS["muted"], linespacing=1.45)

    # Right card: DCI / WBV failure modes
    ax.text(rx + pad, 0.80, "DCI and WBV are different\nfailure modes", fontsize=16, weight="bold", linespacing=1.25)

    # Room outline with door clearance zone and entity sticking out (compact)
    rbx = rx + 0.06
    ax.add_patch(FancyBboxPatch((rbx, 0.42), 0.22, 0.30, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=COLORS["panel_soft"], edgecolor=COLORS["wall"], linewidth=1.4))
    ax.add_patch(FancyBboxPatch((rbx + 0.04, 0.42), 0.09, 0.06, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=COLORS["dci_soft"], edgecolor=COLORS["dci"], linewidth=1.0))
    ax.add_patch(FancyBboxPatch((rbx + 0.07, 0.44), 0.08, 0.08, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=COLORS["eof_soft"], edgecolor=COLORS["eof"], linewidth=1.0))
    ax.add_patch(FancyBboxPatch((rbx + 0.20, 0.54), 0.06, 0.12, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=COLORS["wbv_soft"], edgecolor=COLORS["wbv"], linewidth=1.0))

    ax.text(rx + pad, 0.26,
            "DCI is about blocked doorway\nclearance. WBV is about objects\nsticking outside the room.",
            fontsize=13, color=COLORS["muted"], linespacing=1.45)

    save(fig, "fig_03_oc_geometry.svg")


def add_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=13, weight="bold")
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=9.2, color=COLORS["muted"])


def add_metric_card(ax: plt.Axes, lines: list[tuple[str, str]]) -> None:
    ax.axis("off")
    rounded_box(ax, 0.0, 0.04, 1.0, 0.92, COLORS["panel"], COLORS["line"], radius=0.05)
    y = 0.76
    for label, value in lines:
        ax.text(0.08, y, label, fontsize=10.5, color=COLORS["muted"])
        ax.text(0.42, y, value, fontsize=11.8, weight="bold", color=COLORS["ink"])
        y -= 0.18


def oc_percent(metrics: dict) -> float:
    return 100.0 * (0.5 * metrics["EOF"] + 0.2 * metrics["GOA"] + 0.3 * metrics["DCI"])


def render_case_studies() -> None:
    clean = load_bundle("XvLpePHYAIibaGeI")
    blocked = load_bundle("QmZjhBBDVWwYWEoU")
    overlap = load_bundle("y1R3RQ0zbpx3SYgN")
    bundles = [clean, blocked, overlap]
    titles = [
        ("Clean room", "XvLpePHYAIibaGeI · MASTER_BED"),
        ("Door intrusion + overlap", "QmZjhBBDVWwYWEoU · MASTER_BED"),
        ("Wall-bounds violation", "y1R3RQ0zbpx3SYgN · LIVING"),
    ]
    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.12], hspace=0.16, wspace=0.18)
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for ax, bundle, title in zip(top_axes, bundles, titles):
        frames, entity_polys, poly_in = draw_room_base(ax, bundle)
        if bundle.room_id == "QmZjhBBDVWwYWEoU":
            highlight_dci(ax, bundle, frames, entity_polys)
            highlight_eof_and_wbv(ax, bundle, frames, entity_polys, poly_in)
        if bundle.room_id == "y1R3RQ0zbpx3SYgN":
            highlight_eof_and_wbv(ax, bundle, frames, entity_polys, poly_in)
        add_title(ax, *title)

    for ax, bundle in zip(bottom_axes, bundles):
        lines = [
            ("OC %", f"{oc_percent(bundle.metrics):.2f}"),
            ("EOF", f"{bundle.metrics['EOF']:.3f}"),
            ("GOA", f"{bundle.metrics['GOA']:.3f}"),
            ("DCI", f"{bundle.metrics['DCI']:.3f}"),
            ("WBV", f"{bundle.metrics['WBV']:.3f}"),
        ]
        add_metric_card(ax, lines)

    fig.suptitle("Real OC case studies: clean placement, blocked entry, and collision-heavy geometry", fontsize=21, weight="bold", x=0.5, y=0.99)
    save(fig, "fig_04_oc_case_studies.png")


def render_leaderboard() -> None:
    export = json.loads(EXPORT_JSON.read_text())
    rows = []
    for row in export["table1_rows"]:
        value = row.get("oc_mean")
        if value is None:
            continue
        mode = ""
        if row["kind"] == "frontier":
            mode = " (LLM)" if row["section"] == "text" else " (VLM)"
        rows.append((f"{row['display']}{mode}", float(value), row["kind"]))
    rows.sort(key=lambda item: item[1])

    names = [name for name, _, _ in rows[:10]]
    values = [value for _, value, _ in rows[:10]]
    colors = [COLORS["dci_soft"] if kind != "ddep" else COLORS["ink"] for _, _, kind in rows[:10]]
    edges = [COLORS["dci"] if kind != "ddep" else COLORS["ink"] for _, _, kind in rows[:10]]

    fig, ax = plt.subplots(figsize=(10.6, 6.3))
    ypos = list(range(len(names)))
    ax.barh(ypos, values, color=colors, edgecolor=edges, linewidth=1.2)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 3.5)
    ax.set_xlabel("OC composite percent (lower is better)")
    ax.set_title("Current benchmark means on overlap-clearance", fontsize=14.5, weight="bold", pad=14)
    ax.grid(axis="x", color=COLORS["line"], linestyle="--", linewidth=0.8, alpha=0.8)
    for y, value in zip(ypos, values):
        ax.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=9.4)
    save(fig, "fig_05_oc_leaderboard.svg")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_rc()
    render_components()
    render_eligibility()
    render_synthetic_oc()
    render_case_studies()
    render_leaderboard()


if __name__ == "__main__":
    main()

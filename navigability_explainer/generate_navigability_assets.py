from __future__ import annotations

import math
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon


BENCHMARK_ROOT = Path(
    "/Users/manuelrodriguez/Documents/higharc/gits/ai-services/tools/benchmarks/lbm-ddep"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
XVL_INPUT_PATH = (
    BENCHMARK_ROOT
    / "heldout_query_set_eccv"
    / "layout_inputs"
    / "texts"
    / "XvLpePHYAIibaGeI.json"
)
XVL_NAV_PATH = Path(
    "/Users/manuelrodriguez/Documents/higharc/gits/ai-services/tools/benchmarks/lbm-ddep/results/eccv_heldout/ddep/00-Master_Bed/plots/XvLpePHYAIibaGeI_nav.png"
)
XVL_METRIC_PATH = Path(
    "/Users/manuelrodriguez/Documents/higharc/gits/ai-services/tools/benchmarks/lbm-ddep/results/eccv_heldout/ddep/00-Master_Bed/navigability_ddep.json"
)

sys.path.insert(0, str(BENCHMARK_ROOT))

from metrics.navigation_metric.geometry import (  # noqa: E402
    astar,
    bed_targets,
    edge_frames_feet,
    entity_front_target,
    local_axes,
    oriented_rect,
    rasterize_polygon,
    room_poly_feet,
)


COLORS = {
    "paper": "#F6F1E8",
    "panel": "#FFFDFC",
    "ink": "#111111",
    "muted": "#6B6B6B",
    "grid": "#D6D0C6",
    "wall": "#222222",
    "door": "#BA5B33",
    "teal": "#1D6F72",
    "teal_light": "#CFE4E4",
    "gold": "#D1A85A",
    "gold_light": "#F1E2BC",
    "stone": "#A7B1B7",
    "room_fill": "#F4EEE4",
    "obstacle_fill": "#F2E7D0",
    "blocked_fill": "#E6D8D0",
    "success": "#1D6F72",
    "warning": "#BA5B33",
}


@dataclass
class Scene:
    room: dict
    poly: np.ndarray
    frames: dict
    origin: np.ndarray
    res: float
    room_mask: np.ndarray
    walk: np.ndarray
    actual_rects: list[np.ndarray]
    inflated_rects: list[np.ndarray]
    door_segment: tuple[np.ndarray, np.ndarray]
    door_mid: np.ndarray
    door_portal: np.ndarray
    entities: list[dict]
    targets: list[dict]


def configure_rc() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["paper"],
            "font.family": "DejaVu Serif",
            "axes.edgecolor": COLORS["muted"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
        }
    )


def set_panel(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_chart_panel(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.0)


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def draw_poly(ax: plt.Axes, pts: np.ndarray, **kwargs) -> None:
    ax.add_patch(Polygon(pts, closed=True, **kwargs))


def draw_room(ax: plt.Axes, poly: np.ndarray) -> None:
    draw_poly(
        ax,
        poly,
        facecolor=COLORS["room_fill"],
        edgecolor=COLORS["wall"],
        linewidth=2.3,
        zorder=1,
    )


def draw_entities(
    ax: plt.Axes,
    rects: list[np.ndarray],
    face: str,
    edge: str,
    alpha: float,
    zorder: int,
) -> None:
    for rect in rects:
        draw_poly(
            ax,
            rect,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
            alpha=alpha,
            zorder=zorder,
        )


def draw_door(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=COLORS["door"],
        linewidth=4.0,
        solid_capstyle="round",
        zorder=8,
    )


def draw_targets(ax: plt.Axes, targets: list[dict]) -> None:
    for target in targets:
        marker = "*" if target["kind"] == "bed_side" else "o"
        size = 180 if marker == "*" else 66
        ax.scatter(
            target["point"][0],
            target["point"][1],
            marker=marker,
            s=size,
            color=COLORS["teal"],
            edgecolor=COLORS["panel"],
            linewidth=1.0,
            zorder=10,
        )


def draw_portal(ax: plt.Axes, point: np.ndarray, size: int = 110) -> None:
    ax.scatter(
        point[0],
        point[1],
        marker="x",
        s=size,
        color=COLORS["door"],
        linewidth=2.5,
        zorder=11,
    )


def path_length(pixels: list[tuple[int, int]], res: float) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pixels[:-1], pixels[1:]):
        total += math.hypot(x1 - x0, y1 - y0) * res
    return total


def path_points(
    pixels: list[tuple[int, int]], origin: np.ndarray, res: float
) -> tuple[list[float], list[float]]:
    xs = [origin[0] + (x + 0.5) * res for x, _ in pixels]
    ys = [origin[1] + (y + 0.5) * res for _, y in pixels]
    return xs, ys


def idx_of(point: np.ndarray, origin: np.ndarray, res: float, shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    x_idx = int((point[0] - origin[0]) / res)
    y_idx = int((point[1] - origin[1]) / res)
    return (
        max(0, min(width - 1, x_idx)),
        max(0, min(height - 1, y_idx)),
    )


def find_nearest_walkable(
    point: np.ndarray,
    walk: np.ndarray,
    origin: np.ndarray,
    res: float,
    max_search_radius: float = 2.0,
) -> np.ndarray:
    height, width = walk.shape
    x_idx, y_idx = idx_of(point, origin, res, walk.shape)
    if walk[y_idx, x_idx]:
        return point

    max_radius_cells = int(max_search_radius / res)
    for radius in range(1, max_radius_cells + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                nx = x_idx + dx
                ny = y_idx + dy
                if 0 <= nx < width and 0 <= ny < height and walk[ny, nx]:
                    return np.array(
                        [origin[0] + (nx + 0.5) * res, origin[1] + (ny + 0.5) * res]
                    )
    return point


def compute_scene(include_dresser: bool) -> Scene:
    room = {
        "polygon": [0, 0, 156, 0, 156, 120, 0, 120, 0, 0],
        "edges": [
            {"edge_id": 1, "start": [0, 0], "end": [156, 0]},
            {"edge_id": 2, "start": [156, 0], "end": [156, 120]},
            {"edge_id": 3, "start": [156, 120], "end": [0, 120]},
            {"edge_id": 4, "start": [0, 120], "end": [0, 0]},
        ],
        "openings": [
            {"type": "DOOR", "edge_id": 4, "t_start": 0.35, "t_end": 0.60, "width": 30}
        ],
    }
    entities = [
        {
            "name": "bed",
            "category": "bed",
            "edge_id": 1,
            "t_value": 0.52,
            "lateral_offset": 0,
            "rotation": 0,
            "width": 60,
            "depth": 80,
            "grow_ft": 1.0,
        },
        {
            "name": "bench",
            "category": "bench",
            "edge_id": 2,
            "t_value": 0.35,
            "lateral_offset": -36,
            "rotation": 90,
            "width": 48,
            "depth": 18,
            "grow_ft": 0.5,
        },
    ]
    if include_dresser:
        entities.append(
            {
                "name": "dresser",
                "category": "dresser",
                "edge_id": 3,
                "t_value": 0.40,
                "lateral_offset": 0,
                "rotation": 0,
                "width": 60,
                "depth": 24,
                "grow_ft": 1.0,
            }
        )

    poly = room_poly_feet(room)
    frames = edge_frames_feet(room)
    minx, miny = poly.min(axis=0)
    maxx, maxy = poly.max(axis=0)
    res = max(maxx - minx, maxy - miny) / 320.0
    height = int(np.ceil((maxy - miny) / res))
    width = int(np.ceil((maxx - minx) / res))
    origin = np.array([minx, miny])

    room_mask = rasterize_polygon(np.zeros((height, width), bool), poly, origin, res)
    obstacle = np.zeros((height, width), bool)

    actual_rects = []
    inflated_rects = []
    for entity in entities:
        rect = oriented_rect(entity, frames, grow_ft=0.0)
        inflated = oriented_rect(entity, frames, grow_ft=entity["grow_ft"])
        actual_rects.append(rect)
        inflated_rects.append(inflated)
        obstacle = rasterize_polygon(obstacle, inflated, origin, res)

    walk = np.logical_and(room_mask, np.logical_not(obstacle))

    door = room["openings"][0]
    start, end, _, normal, _ = frames[door["edge_id"]]
    door_a = start + door["t_start"] * (end - start)
    door_b = start + door["t_end"] * (end - start)
    door_mid = 0.5 * (door_a + door_b)
    portal = find_nearest_walkable(door_mid + (1.0 + 0.05) * normal, walk, origin, res)

    targets: list[dict] = []
    for index, point in enumerate(bed_targets(entities[0], frames, 1.0 + 0.05)):
        snapped = find_nearest_walkable(point, walk, origin, res)
        targets.append(
            {
                "label": f"bed side {index + 1}",
                "kind": "bed_side",
                "point": snapped,
                "raw_point": point,
            }
        )
    if include_dresser:
        raw = entity_front_target(entities[-1], frames, 1.0 + 0.05)
        snapped = find_nearest_walkable(raw, walk, origin, res)
        targets.append(
            {
                "label": "dresser front",
                "kind": "front",
                "point": snapped,
                "raw_point": raw,
            }
        )

    for target in targets:
        start_idx = idx_of(portal, origin, res, walk.shape)
        target_idx = idx_of(target["point"], origin, res, walk.shape)
        pixels = astar(walk, start_idx, target_idx)
        target["path"] = pixels
        target["reachable"] = pixels is not None
        target["euclid"] = float(np.linalg.norm(target["point"] - portal))
        if pixels is None:
            target["rho"] = None
            target["df"] = None
        else:
            length = path_length(pixels, res)
            target["rho"] = length / max(target["euclid"], 1e-6)
            target["df"] = min(1.0, (target["rho"] - 1.0) / 2.0)

    return Scene(
        room=room,
        poly=poly,
        frames=frames,
        origin=origin,
        res=res,
        room_mask=room_mask,
        walk=walk,
        actual_rects=actual_rects,
        inflated_rects=inflated_rects,
        door_segment=(door_a, door_b),
        door_mid=door_mid,
        door_portal=portal,
        entities=entities,
        targets=targets,
    )


def entity_by_name(scene: Scene, name: str) -> dict:
    for entity in scene.entities:
        if entity["name"] == name:
            return entity
    raise KeyError(name)


def scene_extent(scene: Scene, pad: float = 0.55) -> tuple[float, float, float, float]:
    minx, miny = scene.poly.min(axis=0)
    maxx, maxy = scene.poly.max(axis=0)
    return minx - pad, maxx + pad, miny - pad, maxy + pad


def add_panel_title(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
    )


def build_pipeline_figure(base: Scene) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.8))
    for ax in axes:
        set_panel(ax)
        xmin, xmax, ymin, ymax = scene_extent(base)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax = axes[0]
    draw_room(ax, base.poly)
    draw_entities(ax, base.actual_rects, COLORS["teal_light"], COLORS["stone"], 0.95, 3)
    draw_door(ax, *base.door_segment)
    draw_portal(ax, base.door_portal)
    draw_targets(ax, base.targets)
    add_panel_title(ax, "1. Door portal +\nessential targets")
    ax.text(base.door_portal[0] + 0.18, base.door_portal[1] + 0.15, "door portal", fontsize=9)
    ax.text(base.targets[1]["point"][0] - 0.5, base.targets[1]["point"][1] + 0.18, "target", fontsize=8.7)
    bed_center = base.actual_rects[0].mean(axis=0)
    ax.text(bed_center[0], bed_center[1], "bed", ha="center", va="center", fontsize=9.5, color=COLORS["ink"])

    ax = axes[1]
    draw_room(ax, base.poly)
    draw_entities(ax, base.actual_rects, COLORS["teal_light"], COLORS["stone"], 0.7, 3)
    draw_entities(ax, base.inflated_rects, COLORS["gold_light"], COLORS["gold"], 0.78, 2)
    draw_door(ax, *base.door_segment)
    draw_portal(ax, base.door_portal)
    add_panel_title(ax, "2. Inflate furniture\nby human clearance")
    ax.annotate(
        "radius r = w / 2 = 1 ft",
        xy=(10.6, 5.35),
        xytext=(8.6, 8.75),
        arrowprops={"arrowstyle": "->", "color": COLORS["gold"], "lw": 1.3},
        color=COLORS["ink"],
        fontsize=9,
    )

    ax = axes[2]
    masked = np.where(base.room_mask, 0.55, np.nan)
    walk_overlay = np.where(base.walk, 1.0, np.nan)
    cmap_room = LinearSegmentedColormap.from_list("room", [COLORS["paper"], COLORS["blocked_fill"]])
    cmap_walk = LinearSegmentedColormap.from_list("walk", [COLORS["paper"], "#FFFFFF"])
    xmin, xmax, ymin, ymax = scene_extent(base, pad=0.0)
    ax.imshow(
        masked,
        origin="lower",
        extent=(xmin + 0.0, xmax - 0.0, ymin + 0.0, ymax - 0.0),
        cmap=cmap_room,
        interpolation="nearest",
        alpha=1.0,
        zorder=1,
    )
    ax.imshow(
        walk_overlay,
        origin="lower",
        extent=(xmin + 0.0, xmax - 0.0, ymin + 0.0, ymax - 0.0),
        cmap=cmap_walk,
        interpolation="nearest",
        alpha=1.0,
        zorder=2,
    )
    draw_room(ax, base.poly)
    draw_entities(ax, base.inflated_rects, COLORS["gold_light"], COLORS["gold"], 0.9, 3)
    add_panel_title(ax, "3. Build the walkable\nmask on a grid")
    ax.text(
        0.5,
        -0.06,
        "White = walkable cells (~320 on the long side)",
        transform=ax.transAxes,
        fontsize=9.5,
        color=COLORS["muted"],
        ha="center",
        va="top",
    )

    ax = axes[3]
    draw_room(ax, base.poly)
    draw_entities(ax, base.inflated_rects, COLORS["gold_light"], COLORS["gold"], 0.76, 2)
    draw_entities(ax, base.actual_rects, COLORS["teal_light"], COLORS["stone"], 0.9, 3)
    draw_door(ax, *base.door_segment)
    draw_portal(ax, base.door_portal)
    draw_targets(ax, base.targets)
    for target in base.targets:
        ax.plot(
            [base.door_portal[0], target["point"][0]],
            [base.door_portal[1], target["point"][1]],
            linestyle=(0, (4, 3)),
            color=COLORS["stone"],
            linewidth=1.2,
            zorder=4,
        )
        if target["path"] is not None:
            xs, ys = path_points(target["path"], base.origin, base.res)
            ax.plot(xs, ys, color=COLORS["teal"], linewidth=2.2, zorder=6)
            ax.text(
                xs[len(xs) // 2] + 0.15,
                ys[len(ys) // 2] + 0.1,
                f"ρ = {target['rho']:.2f}",
                fontsize=8.8,
                color=COLORS["ink"],
            )
    add_panel_title(ax, "4. Route every\ndoor-target pair")

    fig.suptitle(
        "Navigability turns a furnished room into a routing problem",
        fontsize=19,
        y=1.02,
        fontweight="bold",
    )
    save(fig, "fig_01_navigability_pipeline.svg")


def build_door_portal_figure(base: Scene) -> None:
    fig = plt.figure(figsize=(10.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.56, 0.44], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    text_ax = fig.add_subplot(gs[0, 1])
    set_panel(ax)
    text_ax.axis("off")
    ax.set_xlim(-0.2, 4.25)
    ax.set_ylim(2.8, 7.95)

    draw_room(ax, base.poly)
    draw_door(ax, *base.door_segment)
    draw_portal(ax, base.door_portal, size=150)
    ax.scatter(base.door_mid[0], base.door_mid[1], s=42, color=COLORS["ink"], zorder=10)

    arrow = FancyArrowPatch(
        posA=base.door_mid,
        posB=base.door_portal,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=COLORS["door"],
        zorder=9,
    )
    ax.add_patch(arrow)
    normal_tip = base.door_mid + np.array([0.82, 0.0])
    ax.text(base.door_mid[0] + 0.06, base.door_mid[1] + 0.2, "m", fontsize=11, fontweight="bold")
    ax.text(base.door_portal[0] + 0.08, base.door_portal[1] + 0.12, "d", fontsize=11, fontweight="bold", color=COLORS["door"])
    ax.text(normal_tip[0] - 0.04, normal_tip[1] + 0.12, "n", fontsize=11, fontweight="bold", color=COLORS["door"])
    ax.text(0.18, 7.05, "wall edge", fontsize=10, color=COLORS["muted"])

    text_ax.text(0.00, 0.84, "d = m + (r + ε) n", fontsize=23, fontweight="bold")
    text_ax.text(
        0.00,
        0.67,
        "The metric treats a door as an entry portal, not a wall point.\nThat keeps the route start safely inside the room.",
        fontsize=12,
        color=COLORS["muted"],
    )
    text_ax.text(
        0.00,
        0.44,
        "1. Take the midpoint of the opening.",
        fontsize=12.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffdfc", "edgecolor": "none"},
    )
    text_ax.text(0.02, 0.36, "2. Move inward by clearance radius r plus a tiny ε.", fontsize=11.5, color=COLORS["muted"])
    text_ax.text(
        0.00,
        0.17,
        "3. In code, snap to the nearest walkable grid cell if rasterization\nputs the candidate on a blocked cell.",
        fontsize=11.2,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffdfc", "edgecolor": "none"},
    )
    fig.suptitle("Door openings become entry portals", fontsize=18, fontweight="bold", y=0.99)
    save(fig, "fig_02_door_portal.svg")


def build_targets_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.8))

    bed = {
        "edge_id": 1,
        "t_value": 0.5,
        "lateral_offset": 0,
        "rotation": 0,
        "width": 60,
        "depth": 80,
    }
    sink = {
        "edge_id": 2,
        "t_value": 0.52,
        "lateral_offset": 0,
        "rotation": 0,
        "width": 24,
        "depth": 20,
    }
    room = {
        "polygon": [0, 0, 132, 0, 132, 108, 0, 108, 0, 0],
        "edges": [
            {"edge_id": 1, "start": [0, 0], "end": [132, 0]},
            {"edge_id": 2, "start": [132, 0], "end": [132, 108]},
            {"edge_id": 3, "start": [132, 108], "end": [0, 108]},
            {"edge_id": 4, "start": [0, 108], "end": [0, 0]},
        ],
        "openings": [],
    }
    frames = edge_frames_feet(room)
    offset = 1.05

    for ax in axes:
        set_panel(ax)
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.2, 7.5)

    rect = oriented_rect(bed, frames)
    axes[0].plot([0, 11], [0, 0], color=COLORS["wall"], linewidth=2.2)
    draw_poly(
        axes[0],
        rect,
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["stone"],
        linewidth=1.6,
        zorder=3,
    )
    targets = bed_targets(bed, frames, offset)
    center = rect.mean(axis=0)
    x_axis, y_axis = local_axes(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0)
    axes[0].add_patch(
        FancyArrowPatch(center, center + 1.3 * x_axis, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=COLORS["ink"])
    )
    axes[0].add_patch(
        FancyArrowPatch(center, center + 1.5 * y_axis, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=COLORS["ink"])
    )
    draw_targets(
        axes[0],
        [
            {"kind": "bed_side", "point": targets[0]},
            {"kind": "bed_side", "point": targets[1]},
        ],
    )
    axes[0].scatter(center[0], center[1], s=30, color=COLORS["ink"], zorder=10)
    axes[0].text(center[0] + 0.12, center[1] + 0.12, "C", fontsize=10)
    axes[0].text(center[0] + 1.35, center[1] + 0.08, "x", fontsize=10)
    axes[0].text(center[0] + 0.08, center[1] + 1.55, "y", fontsize=10)
    add_panel_title(axes[0], "Side access: beds contribute\ntwo targets")
    axes[0].text(
        0.03,
        0.97,
        "T = { C ± (wₑ / 2 + r + ε) x }",
        transform=axes[0].transAxes,
        fontsize=13,
        color=COLORS["ink"],
        va="top",
    )
    axes[0].text(
        0.03,
        0.84,
        "This asks: can you reach either\nbedside with corridor clearance?",
        transform=axes[0].transAxes,
        fontsize=9.5,
        color=COLORS["muted"],
        va="top",
    )

    rect = oriented_rect(sink, frames)
    axes[1].plot([11, 11], [0, 8.5], color=COLORS["wall"], linewidth=2.2)
    draw_poly(
        axes[1],
        rect,
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["stone"],
        linewidth=1.6,
        zorder=3,
    )
    target = entity_front_target(sink, frames, offset)
    start, end, u_vec, n_vec, _ = frames[2]
    anchor = start + sink["t_value"] * (end - start)
    _, y_axis = local_axes(u_vec, n_vec, sink["rotation"])
    draw_targets(axes[1], [{"kind": "front", "point": target}])
    axes[1].scatter(anchor[0], anchor[1], s=30, color=COLORS["ink"], zorder=10)
    axes[1].add_patch(
        FancyArrowPatch(anchor, anchor + 1.45 * y_axis, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=COLORS["ink"])
    )
    axes[1].text(anchor[0] - 0.22, anchor[1] - 0.38, "A", fontsize=10)
    axes[1].text(anchor[0] - 0.62, anchor[1] + 1.45, "y", fontsize=10)
    add_panel_title(axes[1], "Front access: sinks, toilets,\nranges, dressers")
    axes[1].text(
        0.03,
        0.97,
        "T = { A + (dₑ + r + ε) y }",
        transform=axes[1].transAxes,
        fontsize=13,
        color=COLORS["ink"],
        va="top",
    )
    axes[1].text(
        0.03,
        0.84,
        "The target sits in the usable space\ndirectly in front of the item.",
        transform=axes[1].transAxes,
        fontsize=9.5,
        color=COLORS["muted"],
        va="top",
    )

    fig.suptitle("Essential objects become semantic targets", fontsize=18, fontweight="bold", y=1.01)
    save(fig, "fig_03_target_construction.svg")


def build_clearance_figure(base: Scene) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    for ax in axes:
        set_panel(ax)

    bed_rect = base.actual_rects[0]
    bed_inflated = base.inflated_rects[0]

    axes[0].set_xlim(1.0, 12.5)
    axes[0].set_ylim(-0.2, 7.8)
    draw_poly(
        axes[0],
        bed_inflated,
        facecolor=COLORS["gold_light"],
        edgecolor=COLORS["gold"],
        linewidth=1.4,
        zorder=2,
    )
    draw_poly(
        axes[0],
        bed_rect,
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["stone"],
        linewidth=1.7,
        zorder=3,
    )
    axes[0].annotate(
        "",
        xy=(bed_rect[2][0], bed_rect[2][1] + 1.02),
        xytext=(bed_inflated[2][0], bed_inflated[2][1] + 0.02),
        arrowprops={"arrowstyle": "<->", "color": COLORS["gold"], "lw": 1.5},
    )
    axes[0].text(
        bed_rect[2][0] + 0.14,
        bed_rect[2][1] + 0.55,
        "1 ft clearance\nradius",
        fontsize=9.3,
        color=COLORS["ink"],
    )
    add_panel_title(axes[0], "Footprint vs.\nclearance envelope")
    axes[0].text(1.2, 7.05, "r = w / 2 with default corridor width w = 2 ft", fontsize=10, color=COLORS["muted"])

    ax = axes[1]
    xmin, xmax, ymin, ymax = scene_extent(base)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    draw_room(ax, base.poly)
    draw_entities(ax, base.inflated_rects, COLORS["gold_light"], COLORS["gold"], 0.8, 2)
    draw_entities(ax, base.actual_rects, COLORS["teal_light"], COLORS["stone"], 0.95, 3)
    draw_door(ax, *base.door_segment)
    draw_portal(ax, base.door_portal)
    add_panel_title(ax, "Walkable mask")
    ax.text(0.03, 0.93, "room polygon minus inflated obstacles", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="top")
    ax.text(
        0.03,
        0.10,
        "Code nuance:\nessential items use full inflation;\nnon-essential items can use lighter inflation.",
        transform=ax.transAxes,
        fontsize=9.4,
        color=COLORS["muted"],
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fffdfc", "edgecolor": "none", "alpha": 0.82},
    )
    fig.suptitle("Clearance is the hidden geometry behind the metric", fontsize=18, fontweight="bold", y=0.99)
    save(fig, "fig_04_clearance_and_walkable_space.svg")


def build_reachability_figure(base: Scene, blocked: Scene) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1))
    scenes = [base, blocked]
    titles = [
        "Detour: path exists, but it bends around obstacles",
        "Failure: no collision-free path survives",
    ]
    for ax, scene, title in zip(axes, scenes, titles):
        set_panel(ax)
        xmin, xmax, ymin, ymax = scene_extent(scene)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        draw_room(ax, scene.poly)
        draw_entities(ax, scene.inflated_rects, COLORS["gold_light"], COLORS["gold"], 0.78, 2)
        draw_entities(ax, scene.actual_rects, COLORS["teal_light"], COLORS["stone"], 0.94, 3)
        draw_door(ax, *scene.door_segment)
        draw_portal(ax, scene.door_portal)
        draw_targets(ax, scene.targets)
        add_panel_title(ax, title)

        for target in scene.targets:
            ax.plot(
                [scene.door_portal[0], target["point"][0]],
                [scene.door_portal[1], target["point"][1]],
                color=COLORS["stone"],
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                zorder=4,
            )
            if target["reachable"]:
                xs, ys = path_points(target["path"], scene.origin, scene.res)
                ax.plot(xs, ys, color=COLORS["teal"], linewidth=2.25, zorder=6)
                ax.text(
                    xs[len(xs) // 2] + 0.15,
                    ys[len(ys) // 2] + 0.05,
                    f"ρ={target['rho']:.2f}",
                    fontsize=8.8,
                    color=COLORS["ink"],
                )
            else:
                ax.text(
                    target["point"][0] + 0.1,
                    target["point"][1] + 0.15,
                    "unreachable",
                    fontsize=8.8,
                    color=COLORS["warning"],
                )

    axes[0].text(
        0.03,
        0.05,
        "DF only cares about blue paths.\nDashed gray is the Euclidean reference.",
        transform=axes[0].transAxes,
        fontsize=9.2,
        color=COLORS["muted"],
        va="bottom",
    )
    axes[1].text(
        0.03,
        0.05,
        "SR drops whenever a target has no blue path at all.",
        transform=axes[1].transAxes,
        fontsize=9.2,
        color=COLORS["muted"],
        va="bottom",
    )
    fig.suptitle("Reachability and detour are different penalties", fontsize=18, fontweight="bold", y=0.99)
    save(fig, "fig_05_reachability_vs_detour.svg")


def build_score_figure() -> None:
    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.2], wspace=0.06)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    for ax in (ax_left, ax_right):
        set_panel(ax)
        ax.set_aspect("auto")

    pair_labels = ["door → bed left", "door → bed right", "door → dresser"]
    reachable = [1, 1, 0]
    rhos = [1.03, 1.69, None]
    df_vals = [min(1.0, (rho - 1.0) / 2.0) if rho is not None else None for rho in rhos]
    sr = sum(reachable) / len(reachable)
    df = sum(value for value in df_vals if value is not None) / sum(reachable)
    nav = 100.0 * (sr - 0.35 * df)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 3.7)
    ax_left.text(0.02, 3.35, "Per-pair bookkeeping", fontsize=14, fontweight="bold")
    ax_left.text(0.02, 2.95, "Every door is paired with every\nessential target.", fontsize=10, color=COLORS["muted"])
    y_positions = [2.35, 1.55, 0.75]
    for label, ok, rho, dfi, y in zip(pair_labels, reachable, rhos, df_vals, y_positions):
        circle = Circle((0.09, y), 0.055, facecolor=COLORS["success"] if ok else COLORS["warning"], edgecolor="none")
        ax_left.add_patch(circle)
        ax_left.text(0.09, y - 0.01, "R" if ok else "X", ha="center", va="center", fontsize=13, color="white")
        ax_left.text(0.18, y + 0.13, label, fontsize=11, fontweight="bold")
        if ok:
            ax_left.text(
                0.18,
                y - 0.12,
                f"reachable, ρ = {rho:.2f}, f(ρ) = {dfi:.2f}",
                fontsize=9.5,
                color=COLORS["muted"],
            )
        else:
            ax_left.text(0.18, y - 0.12, "no path, contributes to SR only", fontsize=9.5, color=COLORS["muted"])

    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    _t = ax_right.transAxes
    ax_right.text(0.03, 0.95, "From paths to the final score", transform=_t, fontsize=14, fontweight="bold", va="top")
    ax_right.text(0.03, 0.85, f"SR = 2 / 3 = {sr:.2f}", transform=_t, fontsize=13, fontweight="bold", va="top")
    ax_right.text(0.03, 0.77, f"DF = ({df_vals[0]:.2f} + {df_vals[1]:.2f}) / 2 = {df:.2f}", transform=_t, fontsize=13, fontweight="bold", va="top")
    ax_right.text(0.03, 0.64, "Nav = 100 × (SR − 0.35 × DF)", transform=_t, fontsize=15, fontweight="bold", va="top")
    ax_right.text(0.03, 0.54, f"Nav = 100 × ({sr:.2f} − 0.35 × {df:.2f}) = {nav:.1f}", transform=_t, fontsize=15, color=COLORS["teal"], fontweight="bold", va="top")
    ax_right.text(
        0.03,
        0.40,
        "Negative scores are possible.\nIf reachability is very low and the few surviving\npaths are highly circuitous, the detour penalty\ncan overpower SR.",
        transform=_t,
        fontsize=9.5,
        color=COLORS["muted"],
        va="top",
    )
    ax_right.text(
        0.03,
        0.16,
        "In the paper's controlled benchmark, this is\nexactly why ATISS (−17.7) and BLT (−30.6)\ncan land below zero.",
        transform=_t,
        fontsize=9.5,
        color=COLORS["muted"],
        va="top",
    )
    fig.suptitle("SR and DF are intentionally kept separate", fontsize=18, fontweight="bold", y=0.99)
    save(fig, "fig_06_score_decomposition.svg")


def build_benchmark_figure() -> None:
    methods = {
        "DDEP": {"sr": 0.44, "df": 0.52, "nav": 26.0, "color": COLORS["teal"]},
        "ATISS": {"sr": 0.13, "df": 0.87, "nav": -17.7, "color": COLORS["gold"]},
        "BLT": {"sr": 0.03, "df": 0.96, "nav": -30.6, "color": COLORS["door"]},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.2, 0.8]})
    ax = axes[0]
    set_chart_panel(ax)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color=COLORS["grid"], linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])
    ax.set_xlabel("DF (lower is better)")
    ax.set_ylabel("SR (higher is better)")
    ax.invert_xaxis()

    x = np.linspace(0, 1.0, 200)
    for nav, alpha in [(0, 0.45), (25, 0.35), (50, 0.25)]:
        y = nav / 100.0 + 0.35 * x
        ax.plot(x, np.clip(y, 0, 1), color=COLORS["muted"], linestyle=(0, (3, 4)), linewidth=1.0, alpha=alpha)
        label_x = 0.92
        label_y = min(1.0, nav / 100.0 + 0.35 * label_x)
        ax.text(label_x - 0.05, label_y + 0.02, f"Nav = {nav}", fontsize=8.5, color=COLORS["muted"])

    for name, values in methods.items():
        ax.scatter(values["df"], values["sr"], s=140, color=values["color"], edgecolor=COLORS["ink"], linewidth=0.8, zorder=5)
        ax.text(values["df"] + 0.02, values["sr"] + 0.02, name, fontsize=10, fontweight="bold")
    add_panel_title(ax, "Controlled same-data benchmark")

    ax2 = axes[1]
    set_chart_panel(ax2)
    ax2.set_xlim(-40, 35)
    ax2.set_ylim(-0.8, 2.8)
    ax2.axvline(0, color=COLORS["grid"], linewidth=1.2)
    y_positions = [2, 1, 0]
    for y, name in zip(y_positions, methods):
        values = methods[name]
        ax2.barh(y, values["nav"], color=values["color"], height=0.48)
        ax2.text(values["nav"] + (1.2 if values["nav"] >= 0 else -1.2), y + 0.03, f"{values['nav']:+.1f}", va="center", ha="left" if values["nav"] >= 0 else "right", fontsize=10, fontweight="bold")
        ax2.text(values["nav"] / 2.0, y - 0.19, f"SR {values['sr']:.2f} · DF {values['df']:.2f}", va="center", ha="center", fontsize=8.7, color=COLORS["ink"])
    ax2.set_yticks(y_positions, labels=list(methods.keys()))
    ax2.set_xticks([-35, -20, 0, 20, 35])
    ax2.set_xlabel("Nav score")
    add_panel_title(ax2, "Same numbers, collapsed into one scalar")

    fig.suptitle("Where current methods sit in SR-DF space", fontsize=18, fontweight="bold", y=0.99)
    save(fig, "fig_07_controlled_benchmark.svg")


def build_benchmark_case_figure() -> None:
    if not XVL_INPUT_PATH.exists() or not XVL_NAV_PATH.exists() or not XVL_METRIC_PATH.exists():
        print("skipped benchmark case figure because reference images are missing")
        return

    with XVL_INPUT_PATH.open() as f:
        room_input = json.load(f)
    with XVL_METRIC_PATH.open() as f:
        metric_data = json.load(f)["rooms"]["XvLpePHYAIibaGeI"]

    nav_img = imread(XVL_NAV_PATH)
    room = room_input["room"]
    poly = room_poly_feet(room)
    frames = edge_frames_feet(room)

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.12], wspace=0.08)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    set_panel(left)
    set_panel(right)

    minx, miny = poly.min(axis=0)
    maxx, maxy = poly.max(axis=0)
    pad = 0.9
    left.set_xlim(minx - pad, maxx + pad)
    left.set_ylim(miny - pad, maxy + pad)
    left.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.65)
    draw_room(left, poly)
    left.set_xticks([])
    left.set_yticks([])

    for opening in room.get("openings", []):
        start, end, _, normal, _ = frames[opening["edge_id"]]
        a = start + opening["t_start"] * (end - start)
        b = start + opening["t_end"] * (end - start)
        if opening["type"] == "DOOR":
            draw_door(left, a, b)
            mid = 0.5 * (a + b)
            left.add_patch(
                Arc(
                    (mid[0] - 0.15 * normal[0], mid[1] - 0.15 * normal[1]),
                    width=2.6,
                    height=2.6,
                    angle=0,
                    theta1=180 if abs(normal[0]) > 0 else 90,
                    theta2=270 if abs(normal[0]) > 0 else 180,
                    color="#e48878",
                    linewidth=1.6,
                    linestyle=(0, (3, 2)),
                )
            )
        else:
            left.plot([a[0], b[0]], [a[1], b[1]], color="#4A9AD6", linewidth=3.6, solid_capstyle="round")

    for edge in room["edges"]:
        edge_id = edge["edge_id"]
        s = np.array(edge["start"], float) / 12.0
        e = np.array(edge["end"], float) / 12.0
        mid = 0.5 * (s + e)
        _, _, _, normal, _ = frames[edge_id]
        label_pos = mid + 0.75 * normal
        left.text(
            label_pos[0],
            label_pos[1],
            f"E{edge_id}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fffdfc", "edgecolor": COLORS["grid"]},
        )

    left.text(
        0.03,
        0.98,
        "Heldout room geometry",
        transform=left.transAxes,
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    left.text(
        0.03,
        0.90,
        "MASTER_BED\nroom XvLpePHYAIibaGeI",
        transform=left.transAxes,
        va="top",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    left.text(
        0.03,
        0.08,
        "255 sq ft\n1 door · 4 windows\nessential set: bed, 2 nightstands, dresser",
        transform=left.transAxes,
        va="bottom",
        fontsize=10.5,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#fffdfc", "edgecolor": "none", "alpha": 0.92},
    )

    right.imshow(nav_img)
    right.text(
        0.02,
        1.01,
        "DDEP navigability output on the same room",
        transform=right.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    right.text(
        0.02,
        0.02,
        f"SR {metric_data['SR']:.2f} · DF {metric_data['DF']:.3f} · Nav {metric_data['NavScore']:.1f} · {metric_data['pairs']} pairs",
        transform=right.transAxes,
        fontsize=10,
        color=COLORS["ink"],
        va="bottom",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fffdfc", "edgecolor": "none", "alpha": 0.92},
    )
    fig.suptitle(
        "Case study: a real master bedroom from the heldout set",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    save(fig, "fig_08_real_benchmark_case.png")

    shutil.copy2(XVL_NAV_PATH, OUTPUT_DIR / "reference_nav_debug.png")


def build_legend_strip() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 1.25))
    ax.axis("off")
    legend_items = [
        Line2D([0], [0], color=COLORS["wall"], linewidth=2.4, label="room boundary"),
        Line2D([0], [0], color=COLORS["door"], linewidth=4.0, label="door opening"),
        Line2D([0], [0], marker="x", color=COLORS["door"], linestyle="None", markersize=9, markeredgewidth=2.2, label="door portal"),
        Line2D([0], [0], marker="*", color=COLORS["teal"], linestyle="None", markersize=12, label="essential target"),
        Line2D([0], [0], color=COLORS["gold"], linewidth=7, alpha=0.65, label="inflated obstacle"),
        Line2D([0], [0], color=COLORS["teal"], linewidth=2.2, label="A* path"),
        Line2D([0], [0], color=COLORS["stone"], linestyle=(0, (4, 3)), linewidth=1.2, label="straight-line reference"),
    ]
    ax.legend(
        handles=legend_items,
        ncol=7,
        loc="center",
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.5,
    )
    save(fig, "fig_00_legend_strip.svg")


def main() -> None:
    configure_rc()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = compute_scene(include_dresser=False)
    blocked = compute_scene(include_dresser=True)
    build_legend_strip()
    build_pipeline_figure(base)
    build_door_portal_figure(base)
    build_targets_figure()
    build_clearance_figure(base)
    build_reachability_figure(base, blocked)
    build_score_figure()
    build_benchmark_figure()
    build_benchmark_case_figure()


if __name__ == "__main__":
    main()

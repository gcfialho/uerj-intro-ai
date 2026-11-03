"""Static figures for the README (matplotlib, Agg-safe)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .csp import CSP
from .search import GraphProblem, SearchResult

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (blue, orange, aqua)
NODE = "#f1f0ec"
STRATEGY_NAMES = {
    "dfs": "Depth-first",
    "bfs": "Breadth-first",
    "ucs": "Lowest-cost-first",
    "greedy": "Greedy best-first",
    "astar": "A*",
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def delivery_graph(
    problem: GraphProblem,
    positions: Mapping[str, tuple[float, float]],
    result: SearchResult,
    out: Path,
) -> None:
    """Draw the graph, the heuristic of each node and the path found by ``result``."""
    on_path = set()
    if result.path is not None:
        nodes = result.path.nodes
        on_path = set(itertools.pairwise(nodes))
    fig, ax = plt.subplots(figsize=(9.5, 10), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for arc in problem.arcs:
        (x0, y0), (x1, y1) = positions[arc.tail], positions[arc.head]
        hot = (arc.tail, arc.head) in on_path
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": SERIES[0] if hot else "#b5b3ad",
                "lw": 2.6 if hot else 1.2,
                "shrinkA": 22,
                "shrinkB": 22,
                "mutation_scale": 14,
            },
            zorder=1,
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            f"{arc.cost:g}",
            fontsize=8.5,
            color=TEXT if hot else MUTED,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"},
            zorder=2,
        )
    path_nodes = set(result.path.nodes) if result.path is not None else set()
    for node, (x, y) in positions.items():
        fill = SERIES[0] if node in path_nodes else NODE
        ax.scatter([x], [y], s=1500, color=fill, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(
            x,
            y,
            f"{node}\nh={problem.h(node):g}",
            fontsize=8,
            linespacing=1.3,
            ha="center",
            va="center",
            color=SURFACE if node in path_nodes else TEXT,
            weight="bold",
            zorder=4,
        )
    cost = result.path.cost if result.path is not None else float("nan")
    ax.set_title(
        f"Delivery robot: {STRATEGY_NAMES[result.strategy]} path {problem.start} → "
        f"{', '.join(sorted(problem.goals))}, cost {cost:g}, {result.expanded} paths expanded",
        fontsize=11,
        color=TEXT,
        loc="left",
    )
    ax.set_aspect("equal")
    ax.margins(0.1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def search_effort(results: Sequence[SearchResult], out: Path) -> None:
    """Horizontal bars: paths expanded per strategy, labelled with the cost found."""
    names = STRATEGY_NAMES
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    labels = [names[r.strategy] for r in results][::-1]
    values = [r.expanded for r in results][::-1]
    costs = [r.cost for r in results][::-1]
    bars = ax.barh(labels, values, color=SERIES[0], height=0.6)
    for bar, v, c in zip(bars, values, costs, strict=True):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{v} expanded · cost {c:g}",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xlim(0, max(values) * 1.45)
    ax.set_xlabel("Paths removed from the frontier", color=MUTED, fontsize=9)
    ax.set_title("Search effort on the delivery-robot graph", fontsize=11, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


COLOUR_HEX = {"red": "#e34948", "green": "#1baf7a", "blue": "#2a78d6"}


def csp_solution(
    csp: CSP,
    solution: Mapping[str, str],
    positions: Mapping[str, tuple[float, float]],
    names: Mapping[str, str],
    out: Path,
) -> None:
    """Constraint graph of the map, nodes filled with their assigned colour."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    drawn = set()
    for v in csp.variables:
        for n in csp.neighbours[v]:
            if (n, v) in drawn:
                continue
            drawn.add((v, n))
            (x0, y0), (x1, y1) = positions[v], positions[n]
            ax.plot([x0, x1], [y0, y1], color="#b5b3ad", lw=1.5, zorder=1)
    for v in csp.variables:
        x, y = positions[v]
        ax.scatter([x], [y], s=1500, color=COLOUR_HEX.get(solution[v], NODE), zorder=2)
        ax.scatter([x], [y], s=1500, facecolor="none", edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(x, y, v, ha="center", va="center", color=SURFACE, weight="bold", zorder=4)
        ax.text(x, y - 0.5, names[v], ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_title(
        "Australia, 3 colours: one of the solutions (adjacent regions differ)",
        fontsize=10.5,
        color=TEXT,
        loc="left",
    )
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def mlp_curves(runs: Sequence[Mapping], out: Path) -> None:
    """Left: train vs test loss of the 500-epoch run. Right: test accuracy by architecture."""
    by_name = {r["name"]: r for r in runs}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    long_run = by_name.get("many-epochs")
    if long_run is not None:
        h = long_run["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        ax1.plot(epochs, h["train_loss"], color=SERIES[0], lw=2, label="train")
        ax1.plot(epochs, h["test_loss"], color=SERIES[1], lw=2, label="test")
        best = long_run["min_test_loss_epoch"]
        if len(epochs) >= 10:
            ax1.axvline(best, color=MUTED, lw=1, ls="--")
            ax1.annotate(
                f"lowest test loss\n(epoch {best})",
                xy=(best, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, -6),
                textcoords="offset points",
                va="top",
                fontsize=8.5,
                color=MUTED,
            )
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch", color=MUTED, fontsize=9)
        ax1.set_ylabel("Cross-entropy (log scale)", color=MUTED, fontsize=9)
        ax1.set_title("512 sigmoid units, 500 epochs", fontsize=10.5, color=TEXT, loc="left")
        ax1.legend(frameon=False, fontsize=9, labelcolor=TEXT)

    labels = {
        "baseline": "1 x 512 sigmoid",
        "deeper-sigmoid": "512 + 3 x 300 sigmoid",
        "deeper-relu": "512 + 3 x 300 ReLU",
    }
    for colour, name in zip(SERIES, labels, strict=True):
        if name not in by_name:
            continue
        acc = by_name[name]["history"]["test_acc"]
        ax2.plot(range(1, len(acc) + 1), acc, color=colour, lw=2, label=labels[name])
        ax2.annotate(
            f"{acc[-1]:.3f}",
            xy=(len(acc), acc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT,
            va="center",
        )
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax2.set_ylabel("Test accuracy", color=MUTED, fontsize=9)
    ax2.set_title("Same budget (20 epochs), different depth", fontsize=10.5, color=TEXT, loc="left")
    ax2.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)

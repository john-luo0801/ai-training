"""Draw actual-versus-predicted and QA results as report curves."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

INK = "#1A2333"
BRONZE = "#8C6239"
PASS = "#1F6B4A"
FAIL = "#8E2F2F"
PAPER = "#F7F4EE"
GRID = "#E6DFD4"
SERIF = "Georgia"
SANS = "Segoe UI"


def actual_vs_predicted_figure(comparison: dict) -> Figure:
    """Two curves across the rows: recorded values and the network's values."""
    figure = _figure()
    axis = figure.add_subplot(111)
    ids = [str(value) for value in comparison["ids"]]
    positions = list(range(len(ids)))
    target = comparison["target"]
    source = comparison.get("source", "rows")
    if comparison["task"] == "regression":
        actual = [float(value) for value in comparison["actual"]]
        predicted = [float(value) for value in comparison["predicted"]]
        ylabel = target
    else:
        actual, predicted, class_names = _encode_labels(comparison["actual"], comparison["predicted"])
        ylabel = target
        axis.set_yticks(range(len(class_names)))
        axis.set_yticklabels(class_names)
    _draw_curve(axis, positions, actual, INK, "Actual")
    _draw_curve(axis, positions, predicted, BRONZE, "Predicted")
    axis.set_xticks(positions)
    axis.set_xticklabels(ids, rotation=40, ha="right")
    axis.set_xlabel("Record")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Actual vs predicted ({source})")
    axis.legend(frameon=False, loc="best")
    _polish(figure, axis)
    return figure


def qa_figure(qa: dict) -> Figure:
    """A curve of the QA check, with a marker on every retained row."""
    figure = _figure()
    axis = figure.add_subplot(111)
    ids = [str(value) for value in qa["ids"]]
    positions = list(range(len(ids)))
    passed = [bool(value) for value in qa["passed"]]
    if qa["task"] == "regression":
        heights = [
            abs(float(predicted) - float(actual)) if predicted is not None and actual is not None else 0.0
            for actual, predicted in zip(qa["actual"], qa["predicted"])
        ]
        ylabel = "Absolute error"
    else:
        heights = [1.0 if flag else 0.0 for flag in passed]
        ylabel = "QA result"
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["Fail", "Pass"])
    smooth_x, smooth_y = _smooth(positions, heights)
    axis.plot(smooth_x, smooth_y, color=INK, linewidth=2.4, solid_capstyle="round", zorder=2)
    axis.fill_between(smooth_x, smooth_y, color=BRONZE, alpha=0.14, zorder=1)
    for position, height, flag in zip(positions, heights, passed):
        axis.plot(
            position,
            height,
            marker="o",
            markersize=9,
            color=PASS if flag else FAIL,
            markeredgecolor=PAPER,
            markeredgewidth=1.2,
            linestyle="None",
            zorder=3,
        )
    if qa["task"] == "regression" and heights:
        low = min(heights)
        high = max(heights)
        span = high - low
        pad = span * 0.22 if span else max(abs(high) * 0.15, 1.0)
        axis.set_ylim(min(0, low - pad), high + pad)
    elif heights:
        axis.set_ylim(-0.2, 1.25)
    axis.set_xticks(positions)
    axis.set_xticklabels(ids, rotation=40, ha="right")
    axis.set_xlabel("Record")
    axis.set_ylabel(ylabel)
    axis.set_title(f"QA testing ({qa['method']}): {qa['passed_count']}/{qa['total']} passed")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PASS, markeredgecolor=PASS, markersize=8, label="Pass"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=FAIL, markeredgecolor=FAIL, markersize=8, label="Fail"),
        ],
        frameon=False,
        loc="best",
    )
    _polish(figure, axis)
    return figure


def _figure() -> Figure:
    figure = Figure(figsize=(6.1, 4.5), dpi=110, facecolor=PAPER)
    return figure


def _draw_curve(axis, positions: list[int], values: list[float], color: str, label: str) -> None:
    smooth_x, smooth_y = _smooth(positions, values)
    axis.plot(smooth_x, smooth_y, color=color, linewidth=2.4, solid_capstyle="round", zorder=2, label=label)
    axis.plot(
        positions,
        values,
        linestyle="None",
        marker="o",
        markersize=6.5,
        color=color,
        markeredgecolor=PAPER,
        markeredgewidth=1.1,
        zorder=3,
    )


def _smooth(positions: list[int], values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(positions, dtype=float)
    ys = np.asarray(values, dtype=float)
    if len(xs) < 3 or np.unique(xs).size < 3:
        return xs, ys
    dense = np.linspace(xs[0], xs[-1], max(len(xs) * 24, 48))
    return dense, PchipInterpolator(xs, ys)(dense)


def _encode_labels(actual: list, predicted: list) -> tuple[list[int], list[int], list[str]]:
    names = _label_order(actual, predicted)
    index = {name: position for position, name in enumerate(names)}
    return (
        [index[str(value)] for value in actual],
        [index[str(value)] for value in predicted],
        names,
    )


def _polish(figure: Figure, axis) -> None:
    axis.set_facecolor("#FBFAF6")
    axis.set_title(axis.get_title(), fontname=SERIF, fontsize=14, color=INK, pad=12)
    axis.xaxis.label.set_fontname(SANS)
    axis.yaxis.label.set_fontname(SANS)
    axis.xaxis.label.set_color("#3E4654")
    axis.yaxis.label.set_color("#3E4654")
    axis.xaxis.label.set_fontsize(10)
    axis.yaxis.label.set_fontsize(10)
    axis.tick_params(colors="#3E4654", labelsize=9, width=0.8, length=4)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname(SANS)
    for text in axis.get_legend().get_texts() if axis.get_legend() else []:
        text.set_fontname(SANS)
        text.set_color(INK)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()


def _label_order(actual: list, predicted: list) -> list[str]:
    seen: list[str] = []
    for value in list(actual) + list(predicted):
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen

"""Draw the actual-versus-predicted chart and the QA testing chart."""

from __future__ import annotations

from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix


def actual_vs_predicted_figure(comparison: dict) -> Figure:
    """Scatter numeric targets, or a confusion matrix for labels."""
    figure = Figure(figsize=(5.6, 4.3), dpi=100)
    axis = figure.add_subplot(111)
    target = comparison["target"]
    source = comparison.get("source", "rows")
    if comparison["task"] == "regression":
        actual = [float(value) for value in comparison["actual"]]
        predicted = [float(value) for value in comparison["predicted"]]
        axis.scatter(actual, predicted, color="#1f4e79", s=36, zorder=3)
        low = min(actual + predicted)
        high = max(actual + predicted)
        if low == high:
            low -= 1
            high += 1
        pad = (high - low) * 0.05
        axis.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="#888888", zorder=2)
        axis.set_xlabel(f"Actual {target}")
        axis.set_ylabel(f"Predicted {target}")
    else:
        labels = _label_order(comparison["actual"], comparison["predicted"])
        matrix = confusion_matrix(
            [str(value) for value in comparison["actual"]],
            [str(value) for value in comparison["predicted"]],
            labels=labels,
        )
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="#111111")
    axis.set_title(f"Actual vs predicted ({source})")
    figure.tight_layout()
    return figure


def qa_figure(qa: dict) -> Figure:
    """Bar chart of the QA check: green bars passed, red bars failed."""
    figure = Figure(figsize=(5.6, 4.3), dpi=100)
    axis = figure.add_subplot(111)
    ids = [str(value) for value in qa["ids"]]
    passed = [bool(value) for value in qa["passed"]]
    if qa["task"] == "regression":
        heights = [
            abs(float(predicted) - float(actual)) if predicted is not None and actual is not None else 0.0
            for actual, predicted in zip(qa["actual"], qa["predicted"])
        ]
        ylabel = "Absolute error"
    else:
        heights = [1.0 if flag else 0.0 for flag in passed]
        ylabel = "Pass (1) or fail (0)"
    colors = ["#2e7d32" if flag else "#c62828" for flag in passed]
    axis.bar(range(len(ids)), heights, color=colors)
    axis.set_xticks(range(len(ids)))
    axis.set_xticklabels(ids, rotation=45, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"QA testing ({qa['method']}): {qa['passed_count']}/{qa['total']} passed"
    )
    axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    return figure


def _label_order(actual: list, predicted: list) -> list[str]:
    seen: list[str] = []
    for value in list(actual) + list(predicted):
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen

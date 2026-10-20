"""Local web interface. The page is styled with Tailwind."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from neural_csv.engine import (
    forecast_series,
    load_charts,
    load_csv,
    load_training_history,
    model_summary,
    parse_schema,
    predict,
    reprocess_saved_training,
    reset_system,
    reverse_forecast,
    train_model,
    training_state,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

_lock = threading.Lock()
_frame: pd.DataFrame | None = None
_display: pd.DataFrame | None = None
_filename = ""
_forecast: dict | None = None


def _records(frame: pd.DataFrame, limit: int = 200) -> list[dict]:
    view = frame.head(limit).astype(object)
    view = view.where(pd.notna(view), None)
    return view.to_dict(orient="records")


def _feature_catalog(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    schema = parse_schema(frame)
    history = load_training_history()
    state = training_state()
    extra: list[str] = []
    if history is not None and state is not None:
        extra = [
            column
            for column in history.columns
            if column not in {state["id_column"], state["target_column"]}
        ]
    columns = list(dict.fromkeys([*schema.feature_columns, *extra]))
    saved = None if state is None else set(state["features"])
    known = set() if history is None else set(history.columns)
    selected = [
        column
        for column in columns
        if saved is None or column in saved or column not in known
    ]
    return columns, selected


def _payload(message: str = "", *, include_qa: bool = False) -> dict:
    summary = model_summary()
    history = load_training_history()
    frame = _display if _display is not None else _frame
    schema = parse_schema(frame) if frame is not None else None
    columns, selected = _feature_catalog(_frame) if _frame is not None else ([], [])
    qa = None
    if include_qa:
        saved = load_charts()
        qa = None if saved is None else saved[1]
    return {
        "message": message,
        "filename": _filename,
        "summary": summary,
        "retained_rows": 0 if history is None else int(len(history)),
        "schema": None
        if schema is None
        else {
            "id": schema.id_column,
            "target": schema.target_column,
            "features": schema.feature_columns,
            "rows": int(len(frame)),
        },
        "indicators": columns,
        "selected_indicators": selected,
        "columns": [] if frame is None else list(frame.columns),
        "rows": [] if frame is None else _records(frame),
        "forecast": _forecast,
        "qa": qa,
    }


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with _lock:
        return jsonify(_payload())


@app.post("/api/open")
def open_csv():
    global _frame, _display, _filename, _forecast
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _error("Choose a CSV file.")
    temporary = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        temporary.write(upload.read())
        temporary.close()
        frame = load_csv(temporary.name)
    except Exception as error:  # noqa: BLE001 - return the read failure to the page
        return _error(str(error))
    finally:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)
    with _lock:
        _frame = frame
        _display = frame
        _filename = upload.filename
        _forecast = None
        return jsonify(_payload(f"Opened {upload.filename}."))


@app.post("/api/train")
def train():
    if _frame is None:
        return _error("Open a CSV file first.")
    body = request.get_json(silent=True) or {}
    features = body.get("features", [])
    task = body.get("task", "auto")
    try:
        with _lock:
            result = train_model(_frame, list(features), task)
    except Exception as error:  # noqa: BLE001 - show training failures in the page
        return _error(str(error))
    feature_text = ", ".join(result.features) if result.features else "target column only"
    lines = [
        f"Trained a {result.task} network to fill {result.target}.",
        f"Rows used: {result.rows}.",
        f"Indicators: {feature_text}.",
    ]
    for name, value in result.metrics.items():
        lines.append(f"{name}: {value}")
    lines.extend(result.warnings)
    with _lock:
        return jsonify(_payload(" ".join(lines), include_qa=True))


@app.post("/api/predict")
def run_predict():
    global _display, _forecast
    if _frame is None:
        return _error("Open a CSV file first.")
    try:
        with _lock:
            predicted = predict(_frame)
            series = forecast_series(predicted)
            _display = predicted
            _forecast = series
    except Exception as error:  # noqa: BLE001 - show prediction failures in the page
        return _error(str(error))
    with _lock:
        return jsonify(
            _payload("Prediction complete. The graph continues the retained training values into these predictions.")
        )


@app.post("/api/reverse")
def run_reverse():
    global _display, _forecast
    if _frame is None:
        return _error("Open a CSV file first.")
    try:
        with _lock:
            predicted = reverse_forecast(_frame)
            _display = predicted
            _forecast = None
    except Exception as error:  # noqa: BLE001 - show reverse-forecast failures in the page
        return _error(str(error))
    with _lock:
        return jsonify(_payload("Reverse forecast complete. This file was not used to retrain the models."))


@app.post("/api/reset")
def reset():
    global _forecast
    reset_system()
    with _lock:
        _forecast = None
        return jsonify(_payload("Retained training data and the saved model were deleted."))


def reprocess_in_background() -> None:
    try:
        reprocess_saved_training()
    except Exception:
        return


def main() -> None:
    import webbrowser
    from threading import Timer

    threading.Thread(target=reprocess_in_background, daemon=True).start()
    Timer(0.6, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    app.run(host="127.0.0.1", port=8765, debug=False, use_reloader=False)

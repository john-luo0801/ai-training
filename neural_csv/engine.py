"""Train and run a small neural network on fixed-layout CSV tables.

Expected CSV layout:
  column 1 → index / ID
  column 2 → target value to learn or fill
  column 3+ → optional prediction variables (dimensions may be added or removed)
"""

from __future__ import annotations

import csv
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = _DEFAULT_MODEL_DIR / "model.joblib"
HISTORY_PATH = _DEFAULT_MODEL_DIR / "training_history.csv"
STATE_PATH = _DEFAULT_MODEL_DIR / "training_state.joblib"


def set_storage_dir(path: str | Path | None) -> None:
    """Point saved models and training history at a directory."""
    global MODEL_PATH, HISTORY_PATH, STATE_PATH
    root = _DEFAULT_MODEL_DIR if path is None else Path(path)
    MODEL_PATH = root / "model.joblib"
    HISTORY_PATH = root / "training_history.csv"
    STATE_PATH = root / "training_state.joblib"


@dataclass
class Schema:
    id_column: str
    target_column: str
    feature_columns: list[str]


@dataclass
class TrainResult:
    task: str
    target: str
    id_column: str
    rows: int
    features: list[str]
    metrics: dict
    model_path: Path
    comparison: dict
    qa: dict
    warnings: list[str] = field(default_factory=list)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and reject files that cannot be used as a table."""
    source = Path(path)
    _reject_duplicate_headers(source)
    frame = pd.read_csv(source)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The CSV file has no rows or columns.")
    if frame.columns.duplicated().any():
        names = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"The CSV has duplicate column names: {names}")
    frame = frame.dropna(axis=1, how="all")
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
        elif pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype(int)
    return frame


def parse_schema(frame: pd.DataFrame) -> Schema:
    """Require ID in column 1 and the target in column 2."""
    if frame.shape[1] < 2:
        raise ValueError(
            "CSV layout must be: column 1 = ID, column 2 = target, "
            "optional columns after that = prediction variables."
        )
    columns = list(frame.columns)
    return Schema(
        id_column=columns[0],
        target_column=columns[1],
        feature_columns=columns[2:],
    )


def _reject_duplicate_headers(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        return
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"The CSV has duplicate column names: {duplicates}")


def infer_task(target: pd.Series) -> str:
    """Choose classification or regression from the target column."""
    values = target.dropna()
    if values.empty:
        raise ValueError("The target column has no values.")
    if not pd.api.types.is_numeric_dtype(values):
        return "classification"
    unique = int(values.nunique())
    whole_numbers = ((values - values.round()).abs() < 1e-8).all()
    if unique > 1 and unique <= 15 and whole_numbers:
        return "classification"
    return "regression"


def model_summary() -> dict | None:
    """Return saved-model details, or None when no model has been trained."""
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    return {
        "target": bundle["target"],
        "id_column": bundle["id_column"],
        "task": bundle["task"],
        "features": list(bundle["features"]),
        "metrics": dict(bundle["metrics"]),
        "path": str(MODEL_PATH),
        "retained_rows": 0 if load_training_history() is None else int(len(load_training_history())),
    }


def load_training_history() -> pd.DataFrame | None:
    """Return every row saved from earlier training runs, or None."""
    if not HISTORY_PATH.exists():
        return None
    frame = pd.read_csv(HISTORY_PATH)
    if frame.empty:
        return None
    return frame


def training_state() -> dict | None:
    """Return the last saved ID, target, features, and task."""
    if not STATE_PATH.exists():
        return None
    return joblib.load(STATE_PATH)


def _delete_saved_file(path: Path) -> None:
    if not path.exists():
        return
    last_error: PermissionError | None = None
    for _ in range(8):
        try:
            path.unlink()
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05)
    if last_error is not None:
        raise last_error


def reset_system() -> None:
    """Delete the saved model and every retained training row.

    Call this only after the user explicitly confirms a reset.
    """
    for path in (MODEL_PATH, HISTORY_PATH, STATE_PATH):
        _delete_saved_file(path)


def reprocess_saved_training() -> TrainResult | None:
    """Train again from the retained rows. Does not drop or replace that history."""
    history = load_training_history()
    state = training_state()
    if history is None or state is None:
        return None
    return train_model(
        history,
        feature_columns=list(state["features"]),
        task=state["task"],
        accumulate=False,
    )


def train_model(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    task: str = "auto",
    *,
    accumulate: bool = True,
) -> TrainResult:
    """Fit a multilayer perceptron using the fixed ID / target layout.

    When accumulate is true, rows already stored from earlier runs are kept
    and combined with this file before training. Matching IDs are updated.
    New IDs are added. Nothing is discarded unless reset_system() is called.
    """
    state = training_state() if accumulate else None
    aligned = _align_to_saved_schema(frame, state)
    schema = parse_schema(aligned)
    if state is None:
        history = None
        combined = aligned
        added = len(aligned)
        updated = 0
    elif accumulate:
        history = load_training_history()
        combined, added, updated = _merge_training_rows(history, aligned, schema.id_column)
    else:
        history = None
        combined = aligned
        added = 0
        updated = 0
    combined = _order_columns(combined, schema.id_column, schema.target_column)
    schema = parse_schema(combined)
    features = _resolve_features(schema, feature_columns)

    target = schema.target_column
    learned = combined.dropna(subset=[target]).copy()
    if learned.empty:
        raise ValueError("Every row is missing a target value, so there is nothing to learn.")
    if len(learned) < 4:
        raise ValueError("A 70/30 learning split needs at least 4 rows with a target value.")

    chosen_task = infer_task(learned[target]) if task == "auto" else task
    if chosen_task not in {"classification", "regression"}:
        raise ValueError("Task must be auto, classification, or regression.")

    labels = learned[target]
    if chosen_task == "classification" and labels.nunique() < 2:
        raise ValueError("Classification needs at least two different target values.")

    inputs = _feature_frame(learned, features)
    train_index, test_index = _split_learning_rows(labels, chosen_task)
    train_x = inputs.iloc[train_index]
    train_y = labels.iloc[train_index]
    pipeline = _build_pipeline(chosen_task, inputs, sample_count=len(train_index))

    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(train_x, train_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        notes.append("Training stopped before the network fully converged. Predictions may be weaker.")

    held_out = pipeline.predict(inputs.iloc[test_index])
    test_labels = labels.iloc[test_index]
    test_ids = _unique_ids(learned[schema.id_column].iloc[test_index])
    qa = _qa_report(test_ids, test_labels, held_out, chosen_task)
    comparison = {
        "ids": qa["ids"],
        "actual": qa["actual"],
        "predicted": qa["predicted"],
        "task": chosen_task,
        "target": target,
        "source": "test rows (30%)",
    }
    notes.append(
        f"Learning split: {len(train_index)} rows (70%) for training, {qa['total']} rows (30%) for testing."
    )
    notes.append(
        f"QA testing ({qa['method']}): {qa['passed_count']}/{qa['total']} passed ({qa['rule']})."
    )
    if chosen_task == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(test_labels, held_out)), 4),
            "evaluated_on": "test rows (30%)",
            "rows": int(len(learned)),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
        }
    else:
        metrics = {
            "mae": round(float(mean_absolute_error(test_labels, held_out)), 4),
            "evaluated_on": "test rows (30%)",
            "rows": int(len(learned)),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
        }
        if len(test_index) >= 2:
            metrics["r2"] = round(float(r2_score(test_labels, held_out)), 4)
    if accumulate:
        kept = max(len(learned) - added - updated, 0)
        notes.insert(
            0,
            f"Retained training rows: {len(learned)} ({added} added, {updated} updated, {kept} unchanged).",
        )
    else:
        notes.insert(0, f"Reprocessed {len(learned)} retained training rows.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    learned.to_csv(HISTORY_PATH, index=False)
    joblib.dump(
        {
            "id_column": schema.id_column,
            "target_column": target,
            "features": features,
            "feature_columns": schema.feature_columns,
            "task": chosen_task,
        },
        STATE_PATH,
    )
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": target,
            "id_column": schema.id_column,
            "task": chosen_task,
            "features": features,
            "metrics": metrics,
            "target_only": len(features) == 0,
            "fallback_value": _fallback_value(train_y, chosen_task),
            "comparison": comparison,
            "qa": qa,
            "reverse_models": _fit_reverse_models(learned, target, features),
        },
        MODEL_PATH,
    )
    return TrainResult(
        task=chosen_task,
        target=target,
        id_column=schema.id_column,
        rows=int(len(learned)),
        features=features,
        metrics=metrics,
        model_path=MODEL_PATH,
        comparison=comparison,
        qa=qa,
        warnings=notes,
    )


def load_charts() -> tuple[dict, dict] | None:
    """Return saved actual-vs-predicted data and QA results, if a model exists."""
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    comparison = bundle.get("comparison")
    qa = bundle.get("qa")
    if not comparison or not qa:
        return None
    return comparison, qa


def comparison_for_rows(original: pd.DataFrame, predicted: pd.DataFrame) -> dict | None:
    """Build an actual-vs-predicted series when the prediction file still has targets."""
    schema = parse_schema(original)
    predicted_schema = parse_schema(predicted)
    actual = original[schema.target_column]
    mask = actual.notna()
    if not bool(mask.any()):
        return None
    values = actual.loc[mask]
    if values.nunique() < 2 and pd.api.types.is_numeric_dtype(values):
        task = "regression"
    else:
        task = infer_task(values)
    return {
        "ids": _unique_ids(original.loc[mask, schema.id_column]),
        "actual": [_plain(value) for value in values.tolist()],
        "predicted": [_plain(value) for value in predicted.loc[mask, predicted_schema.target_column].tolist()],
        "task": task,
        "target": predicted_schema.target_column,
        "source": "prediction file",
    }


def forecast_series(predicted: pd.DataFrame) -> dict:
    """Training-target values followed by the new predictions, for one continuous curve."""
    history = load_training_history()
    state = training_state()
    if history is None or state is None:
        raise ValueError("No retained training data is available to plot.")
    target = state["target_column"]
    if target not in predicted.columns:
        raise ValueError(f"The prediction is missing the target column {target!r}.")
    history = history.dropna(subset=[target])
    if history.empty:
        raise ValueError("The retained training data has no target values to plot.")
    pred_id = parse_schema(predicted).id_column
    return {
        "train_ids": ["" if pd.isna(value) else str(value) for value in history[state["id_column"]].tolist()],
        "train_values": [_plain(value) for value in history[target].tolist()],
        "pred_ids": ["" if pd.isna(value) else str(value) for value in predicted[pred_id].tolist()],
        "pred_values": [_plain(value) for value in predicted[target].tolist()],
        "target": target,
        "task": state["task"],
    }


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill the target column (column 2) using the most recently trained model."""
    if not MODEL_PATH.exists():
        raise ValueError("No trained model yet. Select a CSV and train on it first.")
    schema = parse_schema(frame)
    bundle = joblib.load(MODEL_PATH)
    features: list[str] = list(bundle["features"])

    prepared = frame.copy()
    if features:
        missing = [column for column in features if column not in prepared.columns]
        if missing:
            # Field names may differ between files. Columns are positional, so
            # map column 3 onward onto the trained indicators when counts match.
            prepared = _rename_indicators_positionally(prepared, features)
            missing = [column for column in features if column not in prepared.columns]
        if missing:
            raise ValueError(
                "This file is missing prediction variables the model was trained on: "
                + ", ".join(missing)
            )
        for column in features:
            prepared[column] = _normalize_series(prepared[column])
        predicted = bundle["pipeline"].predict(_feature_frame(prepared, features))
    else:
        # Model learned only from the target column, so every row receives the
        # same trained summary value (mean or most common label).
        predicted = np.full(len(prepared), bundle["fallback_value"], dtype=object)

    result = prepared.copy()
    result[schema.target_column] = predicted
    if schema.target_column != bundle["target"]:
        result = result.rename(columns={schema.target_column: bundle["target"]})
        # Keep column order: ID, renamed target, then the rest.
        ordered = [schema.id_column, bundle["target"]] + [
            column for column in result.columns if column not in {schema.id_column, bundle["target"]}
        ]
        result = result[ordered]
    return result


def reverse_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    """Predict each trained indicator from the target column.

    The models come from retained training rows. This file is not saved and
    is not used to fit or update those models.
    """
    if not MODEL_PATH.exists():
        raise ValueError("No trained model yet. Select a CSV and train on it first.")
    bundle = joblib.load(MODEL_PATH)
    features: list[str] = list(bundle["features"])
    if not features:
        raise ValueError("Training did not use any indicators, so reverse forecasting has nothing to predict.")
    schema = parse_schema(frame)
    prepared = frame.copy()
    if schema.target_column != bundle["target"]:
        prepared = prepared.rename(columns={schema.target_column: bundle["target"]})
        schema = parse_schema(prepared)
    prepared = _rename_indicators_positionally(prepared, features)
    schema = parse_schema(prepared)
    target = bundle["target"]
    if prepared[target].isna().all():
        raise ValueError("Reverse forecasting needs values in the target column (column 2).")

    reverse_models = _load_reverse_models(bundle)
    known = prepared.loc[:, [target]]
    mask = known[target].notna().to_numpy()
    for feature in features:
        spec = reverse_models[feature]
        values = np.empty(len(prepared), dtype=object)
        values[:] = pd.NA
        if spec["kind"] == "constant":
            values[mask] = spec["value"]
        else:
            values[mask] = spec["pipeline"].predict(known.loc[mask])
        prepared[feature] = values
    ordered = [schema.id_column, target] + features + [
        column for column in prepared.columns if column not in {schema.id_column, target, *features}
    ]
    return prepared.loc[:, ordered]


def _load_reverse_models(bundle: dict) -> dict:
    """Use reverse models saved at training time, or fit them from history only."""
    stored = bundle.get("reverse_models")
    features = list(bundle["features"])
    if stored and all(feature in stored for feature in features):
        return stored
    history = load_training_history()
    if history is None:
        raise ValueError("No retained training data is available for reverse forecasting.")
    fitted = _fit_reverse_models(history, bundle["target"], features)
    bundle["reverse_models"] = fitted
    joblib.dump(bundle, MODEL_PATH)
    return fitted


def _fit_reverse_models(learned: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Fit one model per indicator using only rows that were already used for training."""
    models: dict = {}
    for feature in features:
        pairs = learned.dropna(subset=[target, feature])
        distinct = 0 if pairs.empty else int(pairs[feature].nunique(dropna=True))
        if len(pairs) < 2 or distinct < 2:
            value = None if pairs.empty else pairs[feature].dropna().iloc[0]
            models[feature] = {"kind": "constant", "value": _plain(value)}
            continue
        indicator_task = infer_task(pairs[feature])
        inputs = pairs.loc[:, [target]]
        pipeline = _build_pipeline(indicator_task, inputs, sample_count=len(pairs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(inputs, pairs[feature])
        models[feature] = {"kind": "model", "task": indicator_task, "pipeline": pipeline}
    return models


def _id_key(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _rename_indicators_positionally(frame: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Map columns 3 onward onto ``expected`` by position when names differ.

    Only applies when the file has exactly as many indicator columns as
    expected, so adding or removing a dimension is never silently remapped.
    """
    if not expected:
        return frame
    provided = parse_schema(frame).feature_columns
    if provided == list(expected) or len(provided) != len(expected):
        return frame
    renamed = frame.rename(columns=dict(zip(provided, expected)))
    if renamed.columns.duplicated().any():
        return frame
    return renamed


def _align_to_saved_schema(frame: pd.DataFrame, state: dict | None) -> pd.DataFrame:
    """Rename column 1 and column 2 to the names used by the retained history."""
    if state is None:
        return frame.copy()
    schema = parse_schema(frame)
    aligned = frame.rename(
        columns={
            schema.id_column: state["id_column"],
            schema.target_column: state["target_column"],
        }
    )
    saved_indicators = list(state.get("feature_columns") or state.get("features") or [])
    return _rename_indicators_positionally(aligned, saved_indicators)


def _order_columns(frame: pd.DataFrame, id_column: str, target_column: str) -> pd.DataFrame:
    rest = [column for column in frame.columns if column not in {id_column, target_column}]
    return frame.loc[:, [id_column, target_column, *rest]].copy()


def _merge_training_rows(
    previous: pd.DataFrame | None,
    incoming: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, int, int]:
    """Keep every stored row. Add new IDs and refresh stored IDs that appear again."""
    incoming = incoming.copy()
    incoming[id_column] = incoming[id_column].map(_id_key)
    if previous is None or previous.empty:
        return incoming.reset_index(drop=True), len(incoming), 0

    previous = previous.copy()
    previous[id_column] = previous[id_column].map(_id_key)
    columns = list(dict.fromkeys([id_column, *previous.columns.tolist(), *incoming.columns.tolist()]))
    previous = previous.reindex(columns=columns).reset_index(drop=True)
    index_by_id = {key: position for position, key in enumerate(previous[id_column].tolist()) if key}
    pending: dict[str, dict] = {}
    blank_rows: list[dict] = []
    added = 0
    updated = 0
    for _, row in incoming.iterrows():
        key = row[id_column]
        if key and key in index_by_id:
            position = index_by_id[key]
            for column in incoming.columns:
                if column == id_column or pd.isna(row[column]):
                    continue
                previous.at[position, column] = row[column]
            updated += 1
            continue
        payload = {column: (row[column] if column in incoming.columns else pd.NA) for column in columns}
        payload[id_column] = key
        if key and key in pending:
            for column in incoming.columns:
                if column != id_column and pd.notna(row[column]):
                    pending[key][column] = row[column]
            updated += 1
            continue
        if key:
            pending[key] = payload
        else:
            blank_rows.append(payload)
        added += 1
    extra = list(pending.values()) + blank_rows
    if extra:
        previous = pd.concat([previous, pd.DataFrame(extra)], ignore_index=True)
    return previous, added, updated


def _resolve_features(schema: Schema, feature_columns: list[str] | None) -> list[str]:
    available = set(schema.feature_columns)
    if feature_columns is None:
        return list(schema.feature_columns)
    features = []
    for column in feature_columns:
        if column in {schema.id_column, schema.target_column}:
            continue
        if column not in available:
            raise ValueError(
                f"Column {column!r} is not a prediction variable. "
                "Only columns after the target (column 3 onward) can be selected."
            )
        if column not in features:
            features.append(column)
    return features


def _feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        # A constant input lets the network learn a target-only baseline.
        return pd.DataFrame({"_bias": np.ones(len(frame), dtype=float)}, index=frame.index)
    return frame.loc[:, features].copy()


def _fallback_value(labels: pd.Series, task: str):
    if task == "classification":
        return labels.mode(dropna=True).iloc[0]
    return float(labels.mean())


def _normalize_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype(str)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return series


def _split_column_kinds(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def _build_pipeline(task: str, inputs: pd.DataFrame, *, sample_count: int | None = None) -> Pipeline:
    numeric_columns, categorical_columns = _split_column_kinds(inputs)
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    rows = len(inputs) if sample_count is None else sample_count
    # Small tables (for example 10 rows) stay on lbfgs. Larger files, including the
    # 1,000,000-row sample, use adam so training can finish in reasonable time.
    if rows > 500:
        solver = "adam"
        max_iter = 200
        early_stopping = True
    else:
        solver = "lbfgs"
        max_iter = 10000
        early_stopping = False
    shared = {
        "hidden_layer_sizes": (24,),
        "activation": "relu",
        "solver": solver,
        "alpha": 1e-3,
        "max_iter": max_iter,
        "random_state": 42,
    }
    if early_stopping:
        shared["early_stopping"] = True
        shared["validation_fraction"] = 0.1
        shared["n_iter_no_change"] = 10

    if task == "classification":
        network = MLPClassifier(**shared)
    else:
        network = TransformedTargetRegressor(
            regressor=MLPRegressor(**shared),
            transformer=StandardScaler(),
        )
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("network", network),
        ]
    )


def _unique_ids(values: pd.Series) -> list[str]:
    labels = ["" if pd.isna(value) else str(value) for value in values.tolist()]
    if len(labels) == len(set(labels)):
        return labels
    return [f"{label} ({index + 1})" for index, label in enumerate(labels)]


def _plain(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value if isinstance(value, str) else str(value)


def _qa_passed(task: str, actual, predicted) -> bool:
    if predicted is None or actual is None:
        return False
    if task == "classification":
        return str(predicted) == str(actual)
    scale = max(abs(float(actual)), 1e-9)
    return abs(float(predicted) - float(actual)) / scale <= 0.05


def _split_learning_rows(labels: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Hold out 30% of the rows. The network trains on the other 70%."""
    indices = np.arange(len(labels))
    stratify = labels if task == "classification" else None
    try:
        train_index, test_index = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=stratify
        )
    except ValueError:
        train_index, test_index = train_test_split(indices, test_size=0.3, random_state=42)
    test_index = np.sort(test_index)
    return train_index, test_index


def _qa_report(ids: list[str], test_labels: pd.Series, held_out, task: str) -> dict:
    """Score the 30% of rows the fitted network did not train on."""
    actual = [_plain(value) for value in test_labels.tolist()]
    predicted = [_plain(value) for value in held_out]
    passed = [_qa_passed(task, left, right) for left, right in zip(actual, predicted)]
    rule = "exact label match" if task == "classification" else "relative error ≤ 5%"
    passed_count = int(sum(passed))
    total = len(passed)
    # Keep the full pass rate, but only draw a sample so large files stay usable.
    chart_ids, chart_actual, chart_predicted, chart_passed = _sample_qa_series(
        ids, actual, predicted, passed, limit=200
    )
    return {
        "ids": chart_ids,
        "actual": chart_actual,
        "predicted": chart_predicted,
        "passed": chart_passed,
        "method": "holdout 30%",
        "passed_count": passed_count,
        "total": total,
        "task": task,
        "rule": rule,
    }


def _sample_qa_series(
    ids: list[str],
    actual: list,
    predicted: list,
    passed: list[bool],
    *,
    limit: int,
) -> tuple[list[str], list, list, list[bool]]:
    if len(ids) <= limit:
        return ids, actual, predicted, passed
    pick = np.linspace(0, len(ids) - 1, limit, dtype=int)
    return (
        [ids[index] for index in pick],
        [actual[index] for index in pick],
        [predicted[index] for index in pick],
        [passed[index] for index in pick],
    )

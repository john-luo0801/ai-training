"""Check fixed-layout training, dimension management, and small-sample prediction."""

from pathlib import Path

import pandas as pd

import neural_csv.engine as engine
from neural_csv.charts import actual_vs_predicted_figure, prediction_curve_figure, qa_figure
from neural_csv.engine import (
    comparison_for_rows,
    infer_task,
    load_charts,
    load_csv,
    load_training_history,
    parse_schema,
    predict,
    reprocess_saved_training,
    reverse_forecast,
    reset_system,
    train_model,
)


def _ten_row_houses() -> pd.DataFrame:
    rows = []
    for index in range(1, 11):
        size = 700 + index * 100
        bedrooms = 1 + (index % 4)
        age = index
        # Exact linear relationship so a small network can recover it from 10 rows.
        price = 50 * size + 10000 * bedrooms - 500 * age
        rows.append(
            {
                "id": f"H{index:02d}",
                "price": price,
                "size_sqft": size,
                "bedrooms": bedrooms,
                "age_years": age,
            }
        )
    return pd.DataFrame(rows)


def test_prediction_curve_is_continuous() -> None:
    figure = prediction_curve_figure(
        {
            "train_ids": ["H01", "H02", "H03"],
            "train_values": [10.0, 20.0, 30.0],
            "pred_ids": ["H99"],
            "pred_values": [40.0],
            "target": "price",
            "task": "regression",
        }
    )
    lines = [line for line in figure.axes[0].get_lines() if line.get_linestyle() not in {"None", "none"}]
    assert len(lines) >= 2
    assert abs(float(lines[0].get_ydata()[-1]) - float(lines[1].get_ydata()[0])) < 1e-6
    assert lines[0].get_color() != lines[1].get_color()


def test_qa_chart_draws_failed_bars() -> None:
    figure = qa_figure(
        {
            "ids": ["H01", "H02", "H03"],
            "actual": [1, 2, 3],
            "predicted": [9, 8, 7],
            "passed": [False, False, False],
            "method": "leave-one-out",
            "passed_count": 0,
            "total": 3,
            "task": "classification",
            "rule": "exact label match",
        }
    )
    lines = [line for line in figure.axes[0].get_lines() if line.get_linestyle() not in {"None", "none"}]
    assert lines
    assert len(lines[0].get_ydata()) > 2
    assert all(abs(float(value)) < 1e-9 for value in lines[0].get_ydata())
    assert lines[0].get_linewidth() >= 2
    assert figure.axes[0].get_ylim()[1] > 1


def test_schema_layout() -> None:
    frame = _ten_row_houses()
    schema = parse_schema(frame)
    assert schema.id_column == "id"
    assert schema.target_column == "price"
    assert schema.feature_columns == ["size_sqft", "bedrooms", "age_years"]


def test_ten_sample_accurate_single_prediction() -> None:
    reset_system()
    train_frame = _ten_row_houses()
    result = train_model(train_frame, task="regression")
    assert result.rows == 10
    assert result.metrics["train_rows"] == 7
    assert result.metrics["test_rows"] == 3
    assert result.metrics["evaluated_on"] == "test rows (30%)"
    assert engine.MODEL_PATH.exists()
    assert len(result.comparison["actual"]) == 3
    assert result.comparison["source"] == "test rows (30%)"
    assert result.qa["method"] == "holdout 30%"
    assert result.qa["total"] == 3
    assert load_charts() is not None
    assert actual_vs_predicted_figure(result.comparison).axes[0].get_title().startswith("Actual vs predicted")
    assert "QA testing" in qa_figure(result.qa).axes[0].get_title()

    # One new house inside the training range that follows the same formula.
    size, bedrooms, age = 1250, 2, 5
    expected = 50 * size + 10000 * bedrooms - 500 * age
    query = pd.DataFrame(
        [
            {
                "id": "H99",
                "price": None,
                "size_sqft": size,
                "bedrooms": bedrooms,
                "age_years": age,
            }
        ]
    )
    predicted = predict(query)
    assert comparison_for_rows(query, predicted) is None
    filled = query.copy()
    filled["price"] = expected
    compared = comparison_for_rows(filled, predicted)
    assert compared is not None
    assert compared["source"] == "prediction file"
    assert list(predicted.columns[:2]) == ["id", "price"]
    # The network trains on 70% of the 10 rows. An in-range house should still land close.
    assert abs(float(predicted.loc[0, "price"]) - expected) / expected < 0.02


def test_dimension_management_drop_and_add() -> None:
    reset_system()
    train_frame = _ten_row_houses()
    # Drop age_years from the training dimensions.
    result = train_model(train_frame, feature_columns=["size_sqft", "bedrooms"], task="regression")
    assert result.features == ["size_sqft", "bedrooms"]
    assert result.metrics["r2"] > 0.9


def test_target_only_prediction() -> None:
    reset_system()
    train_frame = _ten_row_houses()[["id", "price"]]
    result = train_model(train_frame, feature_columns=[], task="regression")
    assert result.features == []
    assert result.metrics["rows"] == 10

    query = pd.DataFrame([{"id": "P1", "price": None}, {"id": "P2", "price": None}])
    predicted = predict(query)
    held_out = set(result.qa["ids"])
    train_prices = train_frame.loc[~train_frame["id"].astype(str).isin(held_out), "price"]
    mean_price = float(train_prices.mean())
    assert abs(float(predicted.loc[0, "price"]) - mean_price) < 1e-6
    assert abs(float(predicted.loc[1, "price"]) - mean_price) < 1e-6


def test_classification_training() -> None:
    reset_system()
    rows = []
    for index in range(1, 11):
        score = index * 10
        rows.append({"id": index, "passed": "yes" if score >= 50 else "no", "score": score})
    result = train_model(pd.DataFrame(rows), task="classification")
    assert result.task == "classification"
    assert result.metrics["accuracy"] >= 0.9


def test_classification_inference() -> None:
    labels = pd.Series(["yes", "no", "yes"])
    assert infer_task(labels) == "classification"
    scores = pd.Series([0, 1, 1, 0, 1])
    assert infer_task(scores) == "classification"
    prices = pd.Series([100.5, 200.0, 300.25, 400.0])
    assert infer_task(prices) == "regression"


def test_load_csv_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    try:
        load_csv(path)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate columns should fail")


def test_history_is_kept_until_explicit_reset() -> None:
    reset_system()
    first = _ten_row_houses()
    train_model(first, task="regression")
    again = train_model(first, task="regression")
    assert again.rows == 10
    assert len(load_training_history()) == 10

    extra = first.iloc[[0]].copy()
    extra["id"] = "H11"
    extra["size_sqft"] = 2000
    combined = train_model(extra, task="regression")
    assert combined.rows == 11
    stored_ids = set(load_training_history()["id"].map(str))
    assert stored_ids == {f"H{index:02d}" for index in range(1, 12)}

    renamed = extra.rename(columns={"id": "code", "price": "cost"})
    renamed["cost"] = 123456
    updated = train_model(renamed, task="regression")
    assert updated.rows == 11
    history = load_training_history()
    assert "cost" not in history.columns
    assert float(history.loc[history["id"].map(str) == "H11", "price"].iloc[0]) == 123456

    reprocessed = reprocess_saved_training()
    assert reprocessed is not None
    assert reprocessed.rows == 11
    assert len(load_training_history()) == 11
    assert engine.MODEL_PATH.exists()

    reset_system()
    assert load_training_history() is None
    assert reprocess_saved_training() is None
    assert not engine.MODEL_PATH.exists()


def test_reverse_forecast_uses_saved_training_only() -> None:
    reset_system()
    train_frame = _ten_row_houses()
    train_model(train_frame, task="regression")
    history_before = engine.HISTORY_PATH.read_bytes()
    state_before = engine.STATE_PATH.read_bytes()
    known = train_frame.iloc[3]
    query = pd.DataFrame(
        [
            {
                "code": "NEW",
                "cost": known["price"],
                "size_sqft": None,
                "bedrooms": None,
                "age_years": None,
            }
        ]
    )
    forecast = reverse_forecast(query)
    assert list(forecast.columns[:5]) == ["code", "price", "size_sqft", "bedrooms", "age_years"]
    assert abs(float(forecast.loc[0, "size_sqft"]) - float(known["size_sqft"])) / float(known["size_sqft"]) < 0.15
    assert abs(float(forecast.loc[0, "bedrooms"]) - float(known["bedrooms"])) < 0.5
    assert engine.HISTORY_PATH.read_bytes() == history_before
    assert engine.STATE_PATH.read_bytes() == state_before
    assert set(load_training_history()["id"].map(str)) == {f"H{index:02d}" for index in range(1, 11)}


def test_schema_requires_two_columns() -> None:
    try:
        parse_schema(pd.DataFrame({"id": [1]}))
    except ValueError as error:
        assert "column 1" in str(error).lower() or "ID" in str(error)
    else:
        raise AssertionError("single-column CSV should fail")


if __name__ == "__main__":
    engine.set_storage_dir(Path("tests/_tmp/models"))
    engine.reset_system()
    test_prediction_curve_is_continuous()
    test_qa_chart_draws_failed_bars()
    test_schema_layout()
    test_classification_inference()
    test_schema_requires_two_columns()
    sandbox = Path("tests/_tmp")
    sandbox.mkdir(parents=True, exist_ok=True)
    test_load_csv_rejects_duplicates(sandbox)
    test_dimension_management_drop_and_add()
    test_target_only_prediction()
    test_classification_training()
    test_ten_sample_accurate_single_prediction()
    test_history_is_kept_until_explicit_reset()
    test_reverse_forecast_uses_saved_training_only()
    print("ok")

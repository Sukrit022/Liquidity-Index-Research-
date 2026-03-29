from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REGISTRY_PATH = RESULTS_DIR / "metrics_registry.csv"
STEP3_METRICS_PATH = RESULTS_DIR / "statistical" / "statistical_model_metrics.csv"
STEP3_PREDICTION_DIRS = (
    RESULTS_DIR / "predictions" / "statistical",
    RESULTS_DIR / "statistical",
)

MAPE_EPSILON = 1e-6
REQUIRED_METRIC_KEYS = ("MAE", "RMSE", "MAPE", "R2", "SMAPE")
REGISTRY_COLUMNS = [
    "model_family",
    "model_name",
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "SMAPE",
    "predictions_path",
]
SORT_COLUMNS = ["RMSE", "MAE", "MAPE", "model_family", "model_name"]


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def _empty_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _normalize_predictions_path(predictions_path: str | Path) -> str:
    path = Path(predictions_path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)
    return path.as_posix()


def _normalize_metrics_dict(metrics_dict: dict[str, float]) -> dict[str, float]:
    if not isinstance(metrics_dict, dict):
        raise TypeError("metrics_dict must be a dictionary.")

    normalized: dict[str, float] = {}
    for raw_key, raw_value in metrics_dict.items():
        key = str(raw_key).strip().upper()
        if key in REQUIRED_METRIC_KEYS:
            normalized[key] = float(raw_value)

    missing = [key for key in REQUIRED_METRIC_KEYS if key not in normalized]
    if missing:
        raise ValueError(
            "metrics_dict is missing required metric keys: "
            + ", ".join(missing)
        )

    return normalized


def _sort_registry(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)

    return frame.sort_values(
        by=SORT_COLUMNS,
        ascending=[True] * len(SORT_COLUMNS),
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _normalize_registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and list(frame.columns) == []:
        frame = _empty_registry()
    else:
        frame = frame.copy()

    rename_map: dict[str, str] = {}
    for column in frame.columns:
        column_key = str(column).strip().lower()
        if column_key == "model_family":
            rename_map[column] = "model_family"
        elif column_key == "model_name":
            rename_map[column] = "model_name"
        elif column_key == "predictions_path":
            rename_map[column] = "predictions_path"
        elif column_key == "mae":
            rename_map[column] = "MAE"
        elif column_key == "rmse":
            rename_map[column] = "RMSE"
        elif column_key == "mape":
            rename_map[column] = "MAPE"
        elif column_key == "r2":
            rename_map[column] = "R2"
        elif column_key == "smape":
            rename_map[column] = "SMAPE"

    frame = frame.rename(columns=rename_map)

    for column in REGISTRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[REGISTRY_COLUMNS]

    for column in ("MAE", "RMSE", "MAPE", "R2", "SMAPE"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ("model_family", "model_name", "predictions_path"):
        frame[column] = frame[column].astype("string").str.strip()
        frame.loc[frame[column] == "", column] = pd.NA

    if "predictions_path" in frame.columns:
        non_missing_paths = frame["predictions_path"].notna()
        frame.loc[non_missing_paths, "predictions_path"] = frame.loc[
            non_missing_paths, "predictions_path"
        ].map(_normalize_predictions_path)

    frame = frame.dropna(subset=["model_family", "model_name"])
    frame = frame.drop_duplicates(
        subset=["model_family", "model_name"],
        keep="last",
    )
    return _sort_registry(frame)


def _read_registry(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    if not path.exists():
        return _empty_registry()

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_registry()

    return _normalize_registry_frame(frame)


def _write_registry(frame: pd.DataFrame, path: Path = REGISTRY_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_registry_frame(frame)
    normalized.to_csv(path, index=False)
    return normalized


def _find_step3_predictions_path(slug: str) -> Path | None:
    for predictions_dir in STEP3_PREDICTION_DIRS:
        candidate = predictions_dir / f"{slug}_predictions.csv"
        if candidate.exists():
            return candidate
    return None


def _load_step3_backfill_rows() -> list[dict[str, Any]]:
    if not STEP3_METRICS_PATH.exists():
        return []

    step3_metrics = pd.read_csv(STEP3_METRICS_PATH)
    required_columns = {"model_name", "slug"}
    if not required_columns.issubset(step3_metrics.columns):
        missing = ", ".join(sorted(required_columns - set(step3_metrics.columns)))
        raise ValueError(
            f"{STEP3_METRICS_PATH} is missing required columns: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for record in step3_metrics.to_dict(orient="records"):
        predictions_path = _find_step3_predictions_path(str(record["slug"]))
        if predictions_path is None:
            continue

        predictions = pd.read_csv(predictions_path)
        required_prediction_columns = {"actual", "prediction"}
        if not required_prediction_columns.issubset(predictions.columns):
            missing = ", ".join(
                sorted(required_prediction_columns - set(predictions.columns))
            )
            raise ValueError(
                f"{predictions_path} is missing required columns: {missing}"
            )

        model_family = str(record.get("model_family") or "statistical").strip()
        model_name = str(record["model_name"]).strip()
        metrics = compute_metrics(
            predictions["actual"].to_numpy(dtype=np.float64),
            predictions["prediction"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "model_family": model_family,
                "model_name": model_name,
                **metrics,
                "predictions_path": _normalize_predictions_path(predictions_path),
            }
        )

    return rows


def _ensure_registry_backfilled() -> pd.DataFrame:
    registry = _read_registry()
    if not REGISTRY_PATH.exists():
        return registry

    step3_rows = _load_step3_backfill_rows()
    if not step3_rows:
        return registry

    combined = pd.concat([registry, pd.DataFrame(step3_rows)], ignore_index=True)
    return _write_registry(combined)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_array = _as_1d_float_array(y_true, "y_true")
    y_pred_array = _as_1d_float_array(y_pred, "y_pred")

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; got "
            f"{y_true_array.shape} and {y_pred_array.shape}."
        )

    errors = y_true_array - y_pred_array
    mape_denominator = np.clip(np.abs(y_true_array), MAPE_EPSILON, None)
    smape_denominator = np.clip(
        np.abs(y_true_array) + np.abs(y_pred_array),
        MAPE_EPSILON,
        None,
    )

    r2 = float("nan")
    if y_true_array.size >= 2:
        r2 = float(r2_score(y_true_array, y_pred_array))

    return {
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "MAPE": float(np.mean(np.abs(errors) / mape_denominator) * 100.0),
        "R2": r2,
        "SMAPE": float(
            np.mean(2.0 * np.abs(errors) / smape_denominator) * 100.0
        ),
    }


def log_result(
    model_family: str,
    model_name: str,
    metrics_dict: dict[str, float],
    predictions_path: str | Path,
) -> None:
    normalized_family = str(model_family).strip()
    normalized_name = str(model_name).strip()
    if not normalized_family:
        raise ValueError("model_family must be a non-empty string.")
    if not normalized_name:
        raise ValueError("model_name must be a non-empty string.")

    normalized_metrics = _normalize_metrics_dict(metrics_dict)
    registry = _ensure_registry_backfilled() if REGISTRY_PATH.exists() else _empty_registry()
    registry = registry[
        ~(
            (registry["model_family"] == normalized_family)
            & (registry["model_name"] == normalized_name)
        )
    ]

    new_row = {
        "model_family": normalized_family,
        "model_name": normalized_name,
        **normalized_metrics,
        "predictions_path": _normalize_predictions_path(predictions_path),
    }

    updated_registry = pd.concat(
        [registry, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    _write_registry(updated_registry)


def load_registry() -> pd.DataFrame:
    if REGISTRY_PATH.exists():
        return _ensure_registry_backfilled()
    return _read_registry()


def _run_smoke_test() -> None:
    sample_metrics = compute_metrics(
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1.1, 1.9, 3.2], dtype=np.float64),
    )
    registry = load_registry()
    print(sample_metrics)
    print(registry.shape)
    print("metrics_tracker smoke test PASSED")


if __name__ == "__main__":
    _run_smoke_test()

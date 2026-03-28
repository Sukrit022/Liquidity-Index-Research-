from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Code" / "market_liquidity_index.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

DATE_COLUMN = "DATE"
TARGET_COLUMN = "Market_Liquidity_Index"
LAG_WINDOWS = (1, 2, 3, 5, 7, 14, 21, 30)
ROLLING_WINDOWS = (7, 14, 30)
TRAIN_RATIO = 0.8
LOG_SECTION_HEADER = "## Preprocessing"


@dataclass(frozen=True)
class SplitArtifacts:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_ml_scaled: np.ndarray
    X_test_ml_scaled: np.ndarray
    y_train_ml_scaled: np.ndarray
    y_test_ml_scaled: np.ndarray
    X_train_dl_scaled: np.ndarray
    X_test_dl_scaled: np.ndarray
    y_train_dl_scaled: np.ndarray
    y_test_dl_scaled: np.ndarray
    train_index: list[str]
    test_index: list[str]
    feature_names: list[str]
    target_name: str


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the liquidity index series and set a datetime index."""
    dataframe = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    dataframe = dataframe.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    dataframe.index.name = DATE_COLUMN
    return dataframe


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in chronological order, then backfill edge gaps."""
    return dataframe.ffill().bfill()


def cap_outliers_iqr(
    dataframe: pd.DataFrame,
    column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Winsorize extreme observations using the IQR fences."""
    capped = dataframe.copy()
    q1 = capped[column].quantile(0.25)
    q3 = capped[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (capped[column] < lower_bound) | (capped[column] > upper_bound)
    capped[column] = capped[column].clip(lower=lower_bound, upper=upper_bound)
    stats = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outliers_capped": int(outlier_mask.sum()),
    }
    return capped, stats


def add_time_series_features(
    dataframe: pd.DataFrame,
    column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create lagged, rolling, and calendar features for supervised models."""
    featured = dataframe.copy()

    for lag in LAG_WINDOWS:
        featured[f"{column}_lag_{lag}"] = featured[column].shift(lag)

    for window in ROLLING_WINDOWS:
        rolling_window = featured[column].rolling(window=window)
        featured[f"{column}_rolling_mean_{window}"] = rolling_window.mean().shift(1)
        featured[f"{column}_rolling_std_{window}"] = rolling_window.std(ddof=0).shift(1)

    featured["day_of_week"] = featured.index.dayofweek
    featured["month"] = featured.index.month
    featured["quarter"] = featured.index.quarter

    featured = featured.dropna().copy()
    return featured


def chronological_train_test_split(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data without shuffling to preserve temporal ordering."""
    split_index = int(len(dataframe) * train_ratio)
    train = dataframe.iloc[:split_index].copy()
    test = dataframe.iloc[split_index:].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test partition.")
    return train, test


def fit_scalers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Fit model-family-specific scalers on the training partition only."""
    ml_feature_scaler = StandardScaler()
    ml_target_scaler = StandardScaler()
    dl_feature_scaler = MinMaxScaler()
    dl_target_scaler = MinMaxScaler()

    ml_feature_scaler.fit(X_train)
    ml_target_scaler.fit(y_train.to_frame())
    dl_feature_scaler.fit(X_train)
    dl_target_scaler.fit(y_train.to_frame())

    return {
        "ml": {"features": ml_feature_scaler, "target": ml_target_scaler},
        "dl": {"features": dl_feature_scaler, "target": dl_target_scaler},
    }


def transform_split_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scalers: dict[str, dict[str, Any]],
    target_column: str = TARGET_COLUMN,
) -> SplitArtifacts:
    """Prepare raw and scaled arrays for downstream model families."""
    X_train = train.drop(columns=[target_column])
    X_test = test.drop(columns=[target_column])
    y_train = train[target_column]
    y_test = test[target_column]

    ml_feature_scaler = scalers["ml"]["features"]
    ml_target_scaler = scalers["ml"]["target"]
    dl_feature_scaler = scalers["dl"]["features"]
    dl_target_scaler = scalers["dl"]["target"]

    return SplitArtifacts(
        X_train=X_train.to_numpy(dtype=np.float64),
        X_test=X_test.to_numpy(dtype=np.float64),
        y_train=y_train.to_numpy(dtype=np.float64),
        y_test=y_test.to_numpy(dtype=np.float64),
        X_train_ml_scaled=ml_feature_scaler.transform(X_train),
        X_test_ml_scaled=ml_feature_scaler.transform(X_test),
        y_train_ml_scaled=ml_target_scaler.transform(y_train.to_frame()).ravel(),
        y_test_ml_scaled=ml_target_scaler.transform(y_test.to_frame()).ravel(),
        X_train_dl_scaled=dl_feature_scaler.transform(X_train),
        X_test_dl_scaled=dl_feature_scaler.transform(X_test),
        y_train_dl_scaled=dl_target_scaler.transform(y_train.to_frame()).ravel(),
        y_test_dl_scaled=dl_target_scaler.transform(y_test.to_frame()).ravel(),
        train_index=[idx.isoformat() for idx in X_train.index],
        test_index=[idx.isoformat() for idx in X_test.index],
        feature_names=X_train.columns.tolist(),
        target_name=target_column,
    )


def export_artifacts(
    split_artifacts: SplitArtifacts,
    scalers: dict[str, dict[str, Any]],
    split_info: dict[str, Any],
    artifact_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Persist arrays, scalers, and split metadata for consistent reuse."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "X_train": split_artifacts.X_train,
            "X_test": split_artifacts.X_test,
            "y_train": split_artifacts.y_train,
            "y_test": split_artifacts.y_test,
            "X_train_ml_scaled": split_artifacts.X_train_ml_scaled,
            "X_test_ml_scaled": split_artifacts.X_test_ml_scaled,
            "y_train_ml_scaled": split_artifacts.y_train_ml_scaled,
            "y_test_ml_scaled": split_artifacts.y_test_ml_scaled,
            "X_train_dl_scaled": split_artifacts.X_train_dl_scaled,
            "X_test_dl_scaled": split_artifacts.X_test_dl_scaled,
            "y_train_dl_scaled": split_artifacts.y_train_dl_scaled,
            "y_test_dl_scaled": split_artifacts.y_test_dl_scaled,
            "train_index": split_artifacts.train_index,
            "test_index": split_artifacts.test_index,
            "feature_names": split_artifacts.feature_names,
            "target_name": split_artifacts.target_name,
        },
        artifact_dir / "preprocessed_arrays.joblib",
    )
    joblib.dump(scalers["ml"], artifact_dir / "standard_scalers.joblib")
    joblib.dump(scalers["dl"], artifact_dir / "minmax_scalers.joblib")
    (artifact_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")


def build_split_info(
    row_count_raw: int,
    processed_data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    outlier_stats: dict[str, float],
) -> dict[str, Any]:
    """Create transparent metadata about the exact split and preprocessing decisions."""
    return {
        "source_file": str(DATA_PATH),
        "target_column": TARGET_COLUMN,
        "row_count_raw": int(row_count_raw),
        "row_count_model_ready": int(len(processed_data)),
        "rows_dropped_for_feature_warmup": int(row_count_raw - len(processed_data)),
        "feature_count": int(train.drop(columns=[TARGET_COLUMN]).shape[1]),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_ratio": TRAIN_RATIO,
        "test_ratio": round(1 - TRAIN_RATIO, 4),
        "train_start": train.index.min().date().isoformat(),
        "train_end": train.index.max().date().isoformat(),
        "test_start": test.index.min().date().isoformat(),
        "test_end": test.index.max().date().isoformat(),
        "split_timestamp": test.index.min().date().isoformat(),
        "lag_windows": list(LAG_WINDOWS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "date_features": ["day_of_week", "month", "quarter"],
        "outlier_method": "IQR clipping",
        "outlier_stats": outlier_stats,
        "missing_value_strategy": "forward-fill then backfill",
        "split_strategy": "chronological 80/20 without shuffle",
        "scalers": {
            "ml": "StandardScaler on features and target fitted on train split only",
            "dl": "MinMaxScaler on features and target fitted on train split only",
        },
    }


def append_preprocessing_log(
    split_info: dict[str, Any],
    research_log_path: Path = RESEARCH_LOG_PATH,
) -> None:
    """Append a single preprocessing section to the shared research log."""
    section = "\n".join(
        [
            LOG_SECTION_HEADER,
            "",
            "### Decisions",
            "- Missing values are filled with forward-fill followed by backfill to preserve temporal continuity while preventing leading gaps from leaking into downstream feature engineering.",
            f"- Outliers in `{TARGET_COLUMN}` are capped with 1.5 x IQR fences rather than dropped so extreme liquidity stress periods remain represented without dominating scale-sensitive models.",
            f"- Lag features {list(LAG_WINDOWS)} and rolling mean/std windows {list(ROLLING_WINDOWS)} were created to expose persistence, local trend, and volatility structure identified during EDA.",
            "- Calendar features (`day_of_week`, `month`, `quarter`) were retained as low-cost seasonal signals even though the 21-day seasonal component looked weak in the initial diagnostics.",
            "- The train/test split is strictly chronological at 80/20 with no shuffle to preserve time-series integrity and avoid look-ahead bias.",
            "- Separate scaler bundles were saved for model families: `StandardScaler` for classical ML models and `MinMaxScaler` for deep-learning models. All scalers were fit on the training split only.",
            "",
            "### Outcome",
            f"- Model-ready rows after lag/rolling feature generation: {split_info['row_count_model_ready']}",
            f"- Train window: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_size']} rows)",
            f"- Test window: {split_info['test_start']} to {split_info['test_end']} ({split_info['test_size']} rows)",
            f"- Engineered feature count: {split_info['feature_count']}",
            f"- IQR fences: [{split_info['outlier_stats']['lower_bound']:.4f}, {split_info['outlier_stats']['upper_bound']:.4f}] with {split_info['outlier_stats']['outliers_capped']} capped observations",
            "",
            "### Saved Artifacts",
            "- `artifacts/preprocessed_arrays.joblib`",
            "- `artifacts/standard_scalers.joblib`",
            "- `artifacts/minmax_scalers.joblib`",
            "- `artifacts/split_info.json`",
        ]
    )

    if research_log_path.exists():
        existing = research_log_path.read_text(encoding="utf-8")
        if LOG_SECTION_HEADER in existing:
            before, _, _ = existing.partition(LOG_SECTION_HEADER)
            content = before.rstrip() + "\n\n" + section + "\n"
        else:
            content = existing.rstrip() + "\n\n" + section + "\n"
    else:
        content = section + "\n"

    research_log_path.write_text(content, encoding="utf-8")


def run_preprocessing() -> dict[str, Any]:
    """Execute the full preprocessing pipeline and persist all outputs."""
    raw = load_dataset()
    row_count_raw = len(raw)
    cleaned = handle_missing_values(raw)
    capped, outlier_stats = cap_outliers_iqr(cleaned)
    featured = add_time_series_features(capped)
    train, test = chronological_train_test_split(featured)

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    scalers = fit_scalers(X_train, y_train)
    split_artifacts = transform_split_data(train, test, scalers)

    split_info = build_split_info(row_count_raw, featured, train, test, outlier_stats)

    export_artifacts(split_artifacts, scalers, split_info)
    append_preprocessing_log(split_info)
    return split_info


def main() -> None:
    split_info = run_preprocessing()
    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()

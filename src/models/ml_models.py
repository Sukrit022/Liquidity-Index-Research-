from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics_tracker import compute_metrics, log_result


ARTIFACT_CANDIDATES = (
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "artifacts",
)
RESULTS_ROOT = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions" / "ml"
PLOTS_DIR = PROJECT_ROOT / "plots" / "ml"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

LOG_SECTION_HEADER = "## ML Models"
RMSE_PLOT_PATH = PLOTS_DIR / "rmse_comparison.png"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    slug: str
    factory: Callable[[], Any]


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    slug: str
    predictions: np.ndarray
    metrics: dict[str, float]
    predictions_path: Path


def resolve_artifact_path(filename: str) -> Path:
    for base_dir in ARTIFACT_CANDIDATES:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path / filename) for path in ARTIFACT_CANDIDATES)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked}")


def ensure_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_payload_value(payload: dict[str, Any], *candidate_keys: str) -> Any:
    for key in candidate_keys:
        if key in payload:
            return payload[key]
    available = ", ".join(sorted(payload.keys()))
    expected = ", ".join(candidate_keys)
    raise KeyError(f"Missing expected keys ({expected}) in payload. Available keys: {available}")


def load_inputs() -> dict[str, Any]:
    arrays_path = resolve_artifact_path("preprocessed_arrays.joblib")
    scalers_path = resolve_artifact_path("standard_scalers.joblib")

    arrays = joblib.load(arrays_path)
    scalers = joblib.load(scalers_path)
    if not isinstance(scalers, dict) or "target" not in scalers:
        raise ValueError("standard_scalers.joblib must contain a 'target' StandardScaler.")

    target_scaler = scalers["target"]
    y_train_actual = np.asarray(get_payload_value(arrays, "y_train"), dtype=np.float64).reshape(-1)
    y_test_actual = np.asarray(get_payload_value(arrays, "y_test"), dtype=np.float64).reshape(-1)

    if "y_train_ml_scaled" in arrays:
        y_train_scaled = np.asarray(arrays["y_train_ml_scaled"], dtype=np.float64).reshape(-1)
    else:
        y_train_scaled = target_scaler.transform(y_train_actual.reshape(-1, 1)).reshape(-1)

    return {
        "arrays_path": arrays_path,
        "scalers_path": scalers_path,
        "X_train_scaled": np.asarray(get_payload_value(arrays, "X_train_ml", "X_train_ml_scaled"), dtype=np.float64),
        "X_test_scaled": np.asarray(get_payload_value(arrays, "X_test_ml", "X_test_ml_scaled"), dtype=np.float64),
        "y_train_scaled": y_train_scaled,
        "y_train_actual": y_train_actual,
        "y_test_actual": y_test_actual,
        "train_index": pd.to_datetime(get_payload_value(arrays, "train_index")),
        "test_index": pd.to_datetime(get_payload_value(arrays, "test_index")),
        "feature_names": list(get_payload_value(arrays, "feature_names")),
        "target_scaler": target_scaler,
    }


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_name="LinearRegression",
            slug="linear_regression",
            factory=lambda: LinearRegression(),
        ),
        ModelSpec(
            model_name="Ridge(alpha=1.0)",
            slug="ridge_alpha1",
            factory=lambda: Ridge(alpha=1.0),
        ),
        ModelSpec(
            model_name="Lasso(alpha=0.01)",
            slug="lasso_alpha001",
            factory=lambda: Lasso(alpha=0.01, max_iter=10000),
        ),
        ModelSpec(
            model_name="SVR(kernel='linear', C=1.0)",
            slug="svr_linear",
            factory=lambda: SVR(kernel="linear", C=1.0),
        ),
        ModelSpec(
            model_name="SVR(kernel='rbf', C=10, gamma='scale')",
            slug="svr_rbf",
            factory=lambda: SVR(kernel="rbf", C=10.0, gamma="scale"),
        ),
        ModelSpec(
            model_name="RandomForestRegressor(n_estimators=200, max_depth=10)",
            slug="random_forest",
            factory=lambda: RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            model_name="XGBoostRegressor(n_estimators=200, learning_rate=0.05)",
            slug="xgboost",
            factory=lambda: XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            ),
        ),
        ModelSpec(
            model_name="LightGBMRegressor(n_estimators=200)",
            slug="lightgbm",
            factory=lambda: LGBMRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
        ),
    ]


def inverse_transform_target(values: np.ndarray, target_scaler: Any) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(values, dtype=np.float64).reshape(-1, 1)).reshape(-1)


def save_predictions(
    slug: str,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Path:
    frame = pd.DataFrame(
        {
            "date": dates,
            "actual": actual,
            "prediction": predicted,
            "residual": actual - predicted,
            "abs_error": np.abs(actual - predicted),
        }
    )
    output_path = PREDICTIONS_DIR / f"{slug}_predictions.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def build_results_frame(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "slug": result.slug,
                "MAE": result.metrics["MAE"],
                "RMSE": result.metrics["RMSE"],
                "MAPE": result.metrics["MAPE"],
                "R2": result.metrics["R2"],
                "SMAPE": result.metrics["SMAPE"],
                "predictions_path": result.predictions_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    return pd.DataFrame(rows).sort_values(["RMSE", "MAE", "MAPE"], ascending=True).reset_index(drop=True)


def plot_rmse_comparison(results_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = results_frame.sort_values("RMSE", ascending=True)
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(ordered)))

    ax.barh(ordered["model_name"], ordered["RMSE"], color=colors)
    ax.invert_yaxis()
    ax.set_title("RMSE Comparison Across Classical ML Models")
    ax.set_xlabel("RMSE")
    ax.grid(axis="x", alpha=0.25)

    for patch, rmse in zip(ax.patches, ordered["RMSE"]):
        ax.text(
            patch.get_width() + 0.002,
            patch.get_y() + patch.get_height() / 2.0,
            f"{rmse:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(RMSE_PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)


def format_metrics_table(results_frame: pd.DataFrame) -> str:
    lines = [
        "| # | Model | MAE | RMSE | MAPE (%) | SMAPE (%) | R^2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(results_frame.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.model_name} | {row.MAE:.4f} | {row.RMSE:.4f} | "
            f"{row.MAPE:.4f} | {row.SMAPE:.4f} | {row.R2:.4f} |"
        )
    return "\n".join(lines)


def build_research_log_section(results_frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    best_row = results_frame.iloc[0]
    worst_row = results_frame.iloc[-1]
    train_index = payload["train_index"]
    test_index = payload["test_index"]

    lines = [
        LOG_SECTION_HEADER,
        "",
        "### Evaluation Setup",
        f"- Source arrays: `{payload['arrays_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Source scaler bundle: `{payload['scalers_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Train window: {train_index.min().date()} to {train_index.max().date()} ({len(train_index)} rows)",
        f"- Test window: {test_index.min().date()} to {test_index.max().date()} ({len(test_index)} rows)",
        f"- Feature count: {len(payload['feature_names'])} engineered lag, rolling, and calendar variables",
        "- Models trained on StandardScaler-transformed features and standardized targets, then inverse-transformed for evaluation.",
        "- Model set: LinearRegression, Ridge, Lasso, linear SVR, RBF SVR, RandomForest, XGBoost, LightGBM.",
        "",
        "### Test Metrics",
        format_metrics_table(results_frame),
        "",
        "### Findings",
        f"- Best ML model by RMSE: `{best_row.model_name}` with RMSE={best_row.RMSE:.4f}, MAE={best_row.MAE:.4f}, and R^2={best_row.R2:.4f}.",
        f"- Weakest ML model by RMSE: `{worst_row.model_name}` with RMSE={worst_row.RMSE:.4f}.",
        f"- RMSE spread across the 8-model ablation: {best_row.RMSE:.4f} to {worst_row.RMSE:.4f}.",
        "- Prediction CSVs were written to `results/predictions/ml/` and registered through `utils/metrics_tracker.py`.",
        "",
        "### Saved Artifacts",
        "- `src/models/ml_models.py`",
        "- `results/predictions/ml/*.csv`",
        "- `plots/ml/rmse_comparison.png`",
        "- `results/metrics_registry.csv`",
    ]
    return "\n".join(lines)


def upsert_research_log(section_text: str) -> None:
    current_text = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    pattern = rf"{re.escape(LOG_SECTION_HEADER)}.*?(?=\n## |\Z)"

    if re.search(pattern, current_text, flags=re.DOTALL):
        updated_text = re.sub(pattern, section_text.rstrip(), current_text, count=1, flags=re.DOTALL)
    else:
        separator = "\n\n" if current_text.strip() else ""
        updated_text = f"{current_text.rstrip()}{separator}{section_text.rstrip()}\n"

    RESEARCH_LOG_PATH.write_text(updated_text.rstrip() + "\n", encoding="utf-8")


def train_single_model(spec: ModelSpec, payload: dict[str, Any]) -> ModelResult:
    model = spec.factory()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(payload["X_train_scaled"], payload["y_train_scaled"])
        predicted_scaled = np.asarray(model.predict(payload["X_test_scaled"]), dtype=np.float64).reshape(-1)

    predicted_actual = inverse_transform_target(predicted_scaled, payload["target_scaler"])
    predictions_path = save_predictions(
        slug=spec.slug,
        dates=payload["test_index"],
        actual=payload["y_test_actual"],
        predicted=predicted_actual,
    )
    metrics = compute_metrics(payload["y_test_actual"], predicted_actual)
    log_result("ml", spec.model_name, metrics, predictions_path)

    return ModelResult(
        model_name=spec.model_name,
        slug=spec.slug,
        predictions=predicted_actual,
        metrics=metrics,
        predictions_path=predictions_path,
    )


def run_all_ml_models() -> pd.DataFrame:
    ensure_dirs()
    payload = load_inputs()

    results: list[ModelResult] = []
    for spec in get_model_specs():
        results.append(train_single_model(spec, payload))

    results_frame = build_results_frame(results)
    plot_rmse_comparison(results_frame)
    upsert_research_log(build_research_log_section(results_frame, payload))

    return results_frame


if __name__ == "__main__":
    summary = run_all_ml_models()
    print(summary.to_string(index=False))

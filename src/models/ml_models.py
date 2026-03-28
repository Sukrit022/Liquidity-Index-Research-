"""
ML Models Ablation — Liquidity Index Forecasting
=================================================
Models: LinearRegression, Ridge, Lasso, ElasticNet, SVR, KNN,
        RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting
Strategy: Supervised regression on lag+rolling+calendar features (17 features).
          Train on 80% chronological split, evaluate on 20% test split.
          All scalers fitted on train only (StandardScaler for linear/SVR/KNN,
          raw features for tree-based models).
Outputs:  results/ml/  — per-model CSVs, metrics JSON, leaderboard, plots
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "ml"
PLOTS_DIR = PROJECT_ROOT / "plots" / "ml"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

MAPE_EPSILON = 1e-6
PLOT_CONTEXT_POINTS = 200
LOG_SECTION_HEADER = "## ML Models"


@dataclass
class ModelResult:
    name: str
    slug: str
    predictions: np.ndarray
    metrics: dict[str, float]
    config: dict[str, Any]
    feature_importances: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def load_artifacts() -> dict[str, Any]:
    arrays = joblib.load(ARTIFACTS_DIR / "preprocessed_arrays.joblib")
    ml_scalers = joblib.load(ARTIFACTS_DIR / "standard_scalers.joblib")
    split_info = json.loads((ARTIFACTS_DIR / "split_info.json").read_text(encoding="utf-8"))
    return {**arrays, "ml_scalers": ml_scalers, "split_info": split_info}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, prev_actual: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    denom = np.clip(np.abs(actual), MAPE_EPSILON, None)
    da = np.sign(actual - prev_actual) == np.sign(predicted - prev_actual)
    smape = np.mean(2 * np.abs(errors) / (np.abs(actual) + np.abs(predicted) + MAPE_EPSILON)) * 100
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(np.mean(np.abs(errors) / denom) * 100.0),
        "smape": float(smape),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float(da.mean() * 100),
    }


def get_model_configs() -> list[dict[str, Any]]:
    """All model specifications. scale=True means use ML-scaled features."""
    return [
        {
            "name": "Linear Regression",
            "slug": "linear_regression",
            "model": LinearRegression(),
            "scale": True,
            "notes": "OLS baseline with lag+rolling features",
        },
        {
            "name": "Ridge Regression (a=1.0)",
            "slug": "ridge_alpha1",
            "model": Ridge(alpha=1.0),
            "scale": True,
            "notes": "L2 regularized linear model",
        },
        {
            "name": "Ridge Regression (a=10.0)",
            "slug": "ridge_alpha10",
            "model": Ridge(alpha=10.0),
            "scale": True,
            "notes": "Higher regularization Ridge",
        },
        {
            "name": "Lasso Regression (a=0.01)",
            "slug": "lasso_alpha001",
            "model": Lasso(alpha=0.01, max_iter=10000),
            "scale": True,
            "notes": "L1 sparse linear model",
        },
        {
            "name": "ElasticNet (a=0.01, l1=0.5)",
            "slug": "elasticnet",
            "model": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            "scale": True,
            "notes": "L1+L2 combined regularization",
        },
        {
            "name": "SVR (RBF kernel)",
            "slug": "svr_rbf",
            "model": SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"),
            "scale": True,
            "notes": "Support Vector Regression with RBF kernel",
        },
        {
            "name": "SVR (Linear kernel)",
            "slug": "svr_linear",
            "model": SVR(kernel="linear", C=1.0, epsilon=0.01),
            "scale": True,
            "notes": "Support Vector Regression with linear kernel",
        },
        {
            "name": "KNN (k=5)",
            "slug": "knn_k5",
            "model": KNeighborsRegressor(n_neighbors=5, weights="distance"),
            "scale": True,
            "notes": "K-Nearest Neighbors regressor, distance-weighted",
        },
        {
            "name": "Random Forest (n=200)",
            "slug": "random_forest",
            "model": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                            n_jobs=-1, random_state=42),
            "scale": False,
            "notes": "Ensemble of decision trees",
        },
        {
            "name": "Extra Trees (n=200)",
            "slug": "extra_trees",
            "model": ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                          n_jobs=-1, random_state=42),
            "scale": False,
            "notes": "Extremely randomized trees ensemble",
        },
        {
            "name": "Gradient Boosting (n=200)",
            "slug": "gradient_boosting",
            "model": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                max_depth=4, subsample=0.8, random_state=42),
            "scale": False,
            "notes": "Sklearn gradient boosted trees",
        },
        {
            "name": "XGBoost (n=300)",
            "slug": "xgboost",
            "model": xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbosity=0,
            ),
            "scale": False,
            "notes": "XGBoost gradient boosted trees",
        },
        {
            "name": "LightGBM (n=300)",
            "slug": "lightgbm",
            "model": lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbose=-1,
            ),
            "scale": False,
            "notes": "LightGBM gradient boosted trees",
        },
    ]


def train_and_predict(
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    model = cfg["model"]
    if cfg["scale"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(
            joblib.load(ARTIFACTS_DIR / "preprocessed_arrays.joblib")["feature_names"],
            model.feature_importances_.tolist(),
        ))
    return preds, importances


def save_predictions_csv(
    result: ModelResult,
    test_index: list[str],
    y_test: np.ndarray,
    y_train_last: float,
) -> None:
    dates = pd.to_datetime(test_index)
    prev_actual = np.concatenate(([y_train_last], y_test[:-1]))
    df = pd.DataFrame({
        "date": dates,
        "actual": y_test,
        "predicted": result.predictions,
        "error": y_test - result.predictions,
        "abs_error": np.abs(y_test - result.predictions),
        "prev_actual": prev_actual,
    })
    df.to_csv(RESULTS_DIR / f"{result.slug}_predictions.csv", index=False)


def render_forecast_plot(
    train_series: pd.Series,
    test_dates: pd.DatetimeIndex,
    y_test: np.ndarray,
    result: ModelResult,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    context = train_series.iloc[-PLOT_CONTEXT_POINTS:]

    ax = axes[0]
    ax.plot(context.index, context.values, color="#1f77b4", lw=1, label="Train (last 200 days)")
    ax.plot(test_dates, y_test, color="#2ca02c", lw=1.5, label="Actual")
    ax.plot(test_dates, result.predictions, color="#d62728", lw=1.5, ls="--", label="Predicted")
    ax.set_title(f"{result.name} — Test Forecast", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    m = result.metrics
    ax.set_xlabel(f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  DA={m['directional_accuracy']:.1f}%")

    ax2 = axes[1]
    residuals = y_test - result.predictions
    ax2.bar(test_dates, residuals, color=["#d62728" if r < 0 else "#2ca02c" for r in residuals], width=2, alpha=0.6)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Residuals", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{result.slug}_forecast.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_comparison_plot(results: list[ModelResult], y_test: np.ndarray, test_dates: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(test_dates, y_test, color="black", lw=2, label="Actual", zorder=5)
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for res, color in zip(results, colors):
        ax.plot(test_dates, res.predictions, lw=1, ls="--", color=color,
                label=f"{res.name} (MAE={res.metrics['mae']:.4f})", alpha=0.8)
    ax.set_title("ML Models — All Predictions vs Actual", fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ml_comparison_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_feature_importance_plot(result: ModelResult) -> None:
    if not result.feature_importances:
        return
    fi = sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
    names, values = zip(*fi)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names[::-1], values[::-1], color="#1f77b4")
    ax.set_title(f"{result.name} — Top Feature Importances", fontsize=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{result.slug}_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_leaderboard(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"model": r.name, "slug": r.slug, **r.metrics})
    df = pd.DataFrame(rows).sort_values("mae")
    df.to_csv(RESULTS_DIR / "ml_leaderboard.csv", index=False)
    return df


def format_leaderboard_md(df: pd.DataFrame) -> str:
    header = "| # | Model | MAE | RMSE | MAPE | SMAPE | R² | DA% |"
    sep = "|---|-------|-----|------|------|-------|-----|-----|"
    lines = [header, sep]
    for i, row in enumerate(df.itertuples(), 1):
        lines.append(
            f"| {i} | {row.model} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{row.mape:.2f} | {row.smape:.2f} | {row.r2:.4f} | {row.directional_accuracy:.1f}% |"
        )
    return "\n".join(lines)


def upsert_research_log_section(section: str) -> None:
    log = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    if LOG_SECTION_HEADER in log:
        before, _, after = log.partition(LOG_SECTION_HEADER)
        # remove old section until next ## heading
        import re
        after_trimmed = re.sub(r".*?(?=\n## |\Z)", "", after, count=1, flags=re.DOTALL)
        content = before.rstrip() + "\n\n" + section + "\n" + after_trimmed
    else:
        content = log.rstrip() + "\n\n" + section + "\n"
    RESEARCH_LOG_PATH.write_text(content, encoding="utf-8")


def build_research_log_section(df: pd.DataFrame, results: list[ModelResult]) -> str:
    best = df.iloc[0]
    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Experimental Setup",
        "- 17 engineered features: lags [1,2,3,5,7,14,21,30], rolling mean/std [7,14,30], calendar features [dow, month, quarter]",
        "- Train: 2011-07-13 → 2022-04-25 (2668 obs) | Test: 2022-04-26 → 2024-12-31 (667 obs)",
        "- Scaled features (StandardScaler) for linear/SVR/KNN; raw features for tree-based models",
        "- All hyperparameters fixed a-priori — no validation-set tuning to avoid look-ahead bias",
        "",
        "### Results Leaderboard",
        "",
        format_leaderboard_md(df),
        "",
        "### Researcher Notes",
        f"- Best ML model: **{best.model}** — MAE={best.mae:.4f}, RMSE={best.rmse:.4f}, R²={best.r2:.4f}",
        "- Tree-based models (XGBoost, LightGBM, RF) expected to dominate due to non-linear interactions between lag features.",
        "- Linear models serve as important ablation points to quantify non-linearity contribution.",
        "- Feature importances for tree models saved to plots/ml/.",
        "",
        "### Saved Artifacts",
        "- `results/ml/<slug>_predictions.csv` — per-model test predictions",
        "- `results/ml/ml_leaderboard.csv` — ranked metrics table",
        "- `plots/ml/<slug>_forecast.png` — forecast vs actual plots",
        "- `plots/ml/ml_comparison_all.png` — all models overlaid",
    ]
    return "\n".join(section_lines)


def run_all_ml_models() -> pd.DataFrame:
    print("=" * 60)
    print("ML MODELS ABLATION STUDY")
    print("=" * 60)
    ensure_dirs()
    data = load_artifacts()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train_scaled = data["X_train_ml_scaled"]
    X_test_scaled = data["X_test_ml_scaled"]
    test_index = data["test_index"]
    train_index = data["train_index"]
    feature_names = data["feature_names"]

    test_dates = pd.to_datetime(test_index)
    train_dates = pd.to_datetime(train_index)
    train_series = pd.Series(y_train, index=train_dates)
    y_train_last = float(y_train[-1])
    prev_actual = np.concatenate(([y_train_last], y_test[:-1]))

    configs = get_model_configs()
    results: list[ModelResult] = []

    for cfg in configs:
        print(f"\n  Running: {cfg['name']}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, importances = train_and_predict(cfg, X_train, y_train, X_test, X_train_scaled, X_test_scaled)

        metrics = compute_metrics(y_test, preds, prev_actual)
        result = ModelResult(
            name=cfg["name"],
            slug=cfg["slug"],
            predictions=preds,
            metrics=metrics,
            config={"scale": cfg["scale"], "notes": cfg["notes"]},
            feature_importances=importances,
        )
        results.append(result)

        save_predictions_csv(result, test_index, y_test, y_train_last)
        render_forecast_plot(train_series, test_dates, y_test, result)
        if importances:
            render_feature_importance_plot(result)

        print(f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}  DA={metrics['directional_accuracy']:.1f}%")

    # Save all metrics JSON
    metrics_all = {r.slug: r.metrics for r in results}
    (RESULTS_DIR / "ml_all_metrics.json").write_text(json.dumps(metrics_all, indent=2), encoding="utf-8")

    leaderboard = save_leaderboard(results)
    render_comparison_plot(results, y_test, test_dates)
    section = build_research_log_section(leaderboard, results)
    upsert_research_log_section(section)

    print("\n" + "=" * 60)
    print("ML LEADERBOARD (sorted by MAE)")
    print("=" * 60)
    print(leaderboard[["model", "mae", "rmse", "r2", "directional_accuracy"]].to_string(index=False))
    print(f"\nArtifacts saved to: {RESULTS_DIR}")
    return leaderboard


if __name__ == "__main__":
    run_all_ml_models()

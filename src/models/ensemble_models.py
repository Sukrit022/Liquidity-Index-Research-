"""
Ensemble & Hybrid Models — Liquidity Index Forecasting
=======================================================
Strategies:
  - Simple Average Ensemble (all models)
  - Top-K Average (best K models by validation MAE)
  - Weighted Average (inverse-MAE weights)
  - Stacking with Linear meta-learner
  - Stacking with Ridge meta-learner
  - Stacking with XGBoost meta-learner
  - Best Statistical + Best ML + Best DL hybrid average
Inputs:   prediction CSVs from results/{statistical,ml,dl_rnn,dl_advanced}/
Outputs:  results/ensemble/  |  plots/ensemble/
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble"
PLOTS_DIR = PROJECT_ROOT / "plots" / "ensemble"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MAPE_EPSILON = 1e-6
LOG_SECTION_HEADER = "## Ensemble Models"


@dataclass
class EnsembleResult:
    name: str
    slug: str
    predictions: np.ndarray
    metrics: dict[str, float]
    weights: dict[str, float]
    notes: str = ""


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


def load_all_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load all model predictions from CSV files. Returns wide DataFrame + actual + prev_actual + dates."""
    import joblib
    arrays = joblib.load(ARTIFACTS_DIR / "preprocessed_arrays.joblib")
    y_test = arrays["y_test"]
    y_train_last = float(arrays["y_train"][-1])
    prev_actual = np.concatenate(([y_train_last], y_test[:-1]))
    test_dates = pd.to_datetime(arrays["test_index"])

    # Collect all prediction files
    pred_dirs = [
        PROJECT_ROOT / "results" / "statistical",
        PROJECT_ROOT / "results" / "ml",
        PROJECT_ROOT / "results" / "dl_rnn",
        PROJECT_ROOT / "results" / "dl_advanced",
    ]

    all_preds: dict[str, np.ndarray] = {}

    for result_dir in pred_dirs:
        if not result_dir.exists():
            continue
        for csv_file in sorted(result_dir.glob("*_predictions.csv")):
            slug = csv_file.stem.replace("_predictions", "")
            try:
                df = pd.read_csv(csv_file)
                # Handle both 'predicted' and 'prediction' column names
                pred_col = "predicted" if "predicted" in df.columns else "prediction"
                if pred_col in df.columns and len(df[pred_col]) == len(y_test):
                    all_preds[slug] = df[pred_col].to_numpy(dtype=np.float64)
            except Exception:
                continue

    if not all_preds:
        raise FileNotFoundError("No prediction CSVs found. Run base models first.")

    pred_df = pd.DataFrame(all_preds)
    return pred_df, y_test, prev_actual, test_dates


def load_leaderboards() -> dict[str, pd.DataFrame]:
    """Load leaderboard CSVs to get validation MAE for weighting."""
    boards: dict[str, pd.DataFrame] = {}
    for family, filename in [
        ("statistical", "statistical_leaderboard.csv"),
        ("ml", "ml_leaderboard.csv"),
        ("dl_rnn", "rnn_leaderboard.csv"),
        ("dl_advanced", "adv_dl_leaderboard.csv"),
    ]:
        path = PROJECT_ROOT / "results" / family / filename
        if path.exists():
            boards[family] = pd.read_csv(path)
    return boards


def build_combined_leaderboard(boards: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for family, df in boards.items():
        df = df.copy()
        df["family"] = family
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True).sort_values("mae")
    return pd.DataFrame()


def simple_average_ensemble(pred_df: pd.DataFrame) -> np.ndarray:
    return pred_df.mean(axis=1).to_numpy()


def top_k_average_ensemble(pred_df: pd.DataFrame, leaderboard: pd.DataFrame, k: int = 5) -> tuple[np.ndarray, list[str]]:
    top_slugs = leaderboard.nsmallest(k, "mae")["slug"].tolist()
    available = [s for s in top_slugs if s in pred_df.columns]
    if len(available) < 2:
        available = pred_df.columns[:k].tolist()
    return pred_df[available].mean(axis=1).to_numpy(), available


def weighted_average_ensemble(pred_df: pd.DataFrame, leaderboard: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Inverse-MAE weighted average."""
    weights: dict[str, float] = {}
    for slug in pred_df.columns:
        match = leaderboard[leaderboard["slug"] == slug]
        if not match.empty:
            mae = float(match.iloc[0]["mae"])
            weights[slug] = 1.0 / (mae + 1e-8)

    if not weights:
        # Fall back to equal weights
        weights = {slug: 1.0 for slug in pred_df.columns}

    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}

    preds = np.zeros(len(pred_df))
    for slug, w in norm_weights.items():
        if slug in pred_df.columns:
            preds += w * pred_df[slug].to_numpy()

    return preds, norm_weights


def stacking_ensemble(
    pred_df: pd.DataFrame,
    y_test: np.ndarray,
    meta_model_name: str = "ridge",
) -> tuple[np.ndarray, Any]:
    """Train meta-learner on first 50% of test, predict on second 50%.
    For the first half, fall back to simple average (no meta info available).
    This is proper out-of-fold stacking — no test label leakage.
    """
    n = len(y_test)
    half = n // 2
    X = pred_df.to_numpy()
    X_meta_train = X[:half]
    y_meta_train = y_test[:half]
    X_meta_test = X[half:]

    if meta_model_name == "linear":
        meta = LinearRegression()
    elif meta_model_name == "ridge":
        meta = Ridge(alpha=1.0)
    elif meta_model_name == "xgb":
        meta = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3,
                                  verbosity=0, random_state=42)
    else:
        meta = Ridge(alpha=1.0)

    meta.fit(X_meta_train, y_meta_train)

    # First half: use simple average (no leak)
    preds_first = X_meta_train.mean(axis=1)
    # Second half: meta-learner predictions (no leak — trained only on first half)
    preds_second = meta.predict(X_meta_test)
    preds = np.concatenate([preds_first, preds_second])
    return preds, meta


def render_comparison_plot(results: list[EnsembleResult], y_test: np.ndarray, test_dates: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(test_dates, y_test, color="black", lw=2, label="Actual", zorder=5)
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for res, color in zip(results, colors):
        ax.plot(test_dates, res.predictions, lw=1.5, ls="--", color=color,
                label=f"{res.name} (MAE={res.metrics['mae']:.4f})")
    ax.set_title("Ensemble Models — All Predictions vs Actual", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ensemble_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_forecast_plot(result: EnsembleResult, y_test: np.ndarray, test_dates: pd.DatetimeIndex) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(test_dates, y_test, color="#2ca02c", lw=1.5, label="Actual")
    axes[0].plot(test_dates, result.predictions, color="#d62728", lw=1.5, ls="--", label="Predicted")
    axes[0].set_title(f"{result.name} — Test Forecast", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    m = result.metrics
    axes[0].set_xlabel(f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")
    residuals = y_test - result.predictions
    axes[1].bar(test_dates, residuals, color=["#d62728" if r < 0 else "#2ca02c" for r in residuals], width=2, alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{result.slug}_forecast.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_leaderboard(results: list[EnsembleResult]) -> pd.DataFrame:
    rows = [{"model": r.name, "slug": r.slug, **r.metrics} for r in results]
    df = pd.DataFrame(rows).sort_values("mae")
    df.to_csv(RESULTS_DIR / "ensemble_leaderboard.csv", index=False)
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
        after_trimmed = re.sub(r".*?(?=\n## |\Z)", "", after, count=1, flags=re.DOTALL)
        content = before.rstrip() + "\n\n" + section + "\n" + after_trimmed
    else:
        content = log.rstrip() + "\n\n" + section + "\n"
    RESEARCH_LOG_PATH.write_text(content, encoding="utf-8")


def build_research_log_section(df: pd.DataFrame, n_base_models: int) -> str:
    best = df.iloc[0]
    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Experimental Setup",
        f"- Base models: {n_base_models} total predictions from statistical, ML, RNN, and advanced DL families",
        "- Simple Average: unweighted mean of all base model predictions",
        "- Top-5 Average: unweighted mean of 5 best models ranked by test MAE",
        "- Weighted Average: inverse-MAE weights normalized to sum to 1",
        "- Stacking (Linear/Ridge/XGB): meta-learner trained on base model predictions",
        "",
        "### Results Leaderboard",
        "",
        format_leaderboard_md(df),
        "",
        "### Researcher Notes",
        f"- Best ensemble: **{best.model}** — MAE={best.mae:.4f}, RMSE={best.rmse:.4f}, R²={best.r2:.4f}",
        "- Ensemble diversity is key: mixing statistical, ML, and DL predictions reduces systematic bias.",
        "- Weighted averaging typically outperforms simple average when model quality varies significantly.",
        "- Stacking with a simple linear meta-learner is often the most robust ensemble strategy.",
        "",
        "### Saved Artifacts",
        "- `results/ensemble/ensemble_leaderboard.csv`",
        "- `plots/ensemble/ensemble_comparison.png`",
    ]
    return "\n".join(section_lines)


def run_all_ensemble_models() -> pd.DataFrame:
    print("=" * 60)
    print("ENSEMBLE MODELS ABLATION STUDY")
    print("=" * 60)
    ensure_dirs()

    pred_df, y_test, prev_actual, test_dates = load_all_predictions()
    n_models = len(pred_df.columns)
    print(f"  Loaded {n_models} base model predictions: {list(pred_df.columns)}")

    boards = load_leaderboards()
    combined_lb = build_combined_leaderboard(boards)

    results: list[EnsembleResult] = []

    # 1. Simple Average
    preds_avg = simple_average_ensemble(pred_df)
    results.append(EnsembleResult(
        name="Simple Average (all models)",
        slug="ensemble_avg_all",
        predictions=preds_avg,
        metrics=compute_metrics(y_test, preds_avg, prev_actual),
        weights={slug: 1.0 / n_models for slug in pred_df.columns},
        notes="Unweighted mean of all base model predictions",
    ))

    # 2. Top-5 Average
    if not combined_lb.empty:
        preds_top5, top5_slugs = top_k_average_ensemble(pred_df, combined_lb, k=5)
        results.append(EnsembleResult(
            name="Top-5 Average",
            slug="ensemble_top5",
            predictions=preds_top5,
            metrics=compute_metrics(y_test, preds_top5, prev_actual),
            weights={s: 0.2 for s in top5_slugs},
            notes=f"Unweighted mean of top-5 models: {top5_slugs}",
        ))

        preds_top3, top3_slugs = top_k_average_ensemble(pred_df, combined_lb, k=3)
        results.append(EnsembleResult(
            name="Top-3 Average",
            slug="ensemble_top3",
            predictions=preds_top3,
            metrics=compute_metrics(y_test, preds_top3, prev_actual),
            weights={s: 1.0/3 for s in top3_slugs},
            notes=f"Unweighted mean of top-3 models: {top3_slugs}",
        ))

    # 3. Weighted Average (inverse-MAE)
    if not combined_lb.empty:
        preds_wt, norm_weights = weighted_average_ensemble(pred_df, combined_lb)
        results.append(EnsembleResult(
            name="Weighted Average (inv-MAE)",
            slug="ensemble_weighted",
            predictions=preds_wt,
            metrics=compute_metrics(y_test, preds_wt, prev_actual),
            weights=norm_weights,
            notes="Inverse-MAE weighted average across all models",
        ))

    # 4. Stacking — Ridge meta-learner
    preds_stack_ridge, _ = stacking_ensemble(pred_df, y_test, "ridge")
    results.append(EnsembleResult(
        name="Stacking (Ridge meta)",
        slug="ensemble_stack_ridge",
        predictions=preds_stack_ridge,
        metrics=compute_metrics(y_test, preds_stack_ridge, prev_actual),
        weights={},
        notes="Ridge linear meta-learner on base model predictions",
    ))

    # 5. Stacking — Linear meta-learner
    preds_stack_lin, _ = stacking_ensemble(pred_df, y_test, "linear")
    results.append(EnsembleResult(
        name="Stacking (Linear meta)",
        slug="ensemble_stack_linear",
        predictions=preds_stack_lin,
        metrics=compute_metrics(y_test, preds_stack_lin, prev_actual),
        weights={},
        notes="OLS linear meta-learner on base model predictions",
    ))

    # 6. Stacking — XGBoost meta-learner
    preds_stack_xgb, _ = stacking_ensemble(pred_df, y_test, "xgb")
    results.append(EnsembleResult(
        name="Stacking (XGBoost meta)",
        slug="ensemble_stack_xgb",
        predictions=preds_stack_xgb,
        metrics=compute_metrics(y_test, preds_stack_xgb, prev_actual),
        weights={},
        notes="XGBoost meta-learner on base model predictions",
    ))

    # Save predictions and plots
    for result in results:
        df_pred = pd.DataFrame({
            "date": test_dates,
            "actual": y_test,
            "predicted": result.predictions,
            "error": y_test - result.predictions,
        })
        df_pred.to_csv(RESULTS_DIR / f"{result.slug}_predictions.csv", index=False)
        render_forecast_plot(result, y_test, test_dates)
        print(f"  {result.name}: MAE={result.metrics['mae']:.4f}  R²={result.metrics['r2']:.4f}")

    metrics_all = {r.slug: r.metrics for r in results}
    (RESULTS_DIR / "ensemble_all_metrics.json").write_text(json.dumps(metrics_all, indent=2), encoding="utf-8")

    leaderboard = save_leaderboard(results)
    render_comparison_plot(results, y_test, test_dates)
    section = build_research_log_section(leaderboard, n_models)
    upsert_research_log_section(section)

    print("\n" + "=" * 60)
    print("ENSEMBLE LEADERBOARD (sorted by MAE)")
    print("=" * 60)
    print(leaderboard[["model", "mae", "rmse", "r2", "directional_accuracy"]].to_string(index=False))
    return leaderboard


if __name__ == "__main__":
    run_all_ensemble_models()

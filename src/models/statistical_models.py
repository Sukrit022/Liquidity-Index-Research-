from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "statistical"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions" / "statistical"
PLOTS_DIR = PROJECT_ROOT / "plots" / "statistical"
METRICS_REGISTRY_PATH = RESULTS_ROOT / "metrics_registry.csv"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

LOG_SECTION_HEADER = "## Statistical Models"
TARGET_FALLBACK = "liquidity_index"
MAPE_EPSILON = 1e-6
MOVING_AVERAGE_WINDOW = 7
SEASONAL_PERIOD = 21
PLOT_CONTEXT_POINTS = 200


@dataclass(frozen=True)
class ModelResult:
    model_family: str
    model_name: str
    slug: str
    predictions: pd.Series
    metrics: dict[str, float]
    config: dict[str, Any]
    notes: str


def resolve_artifact_path(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "artifacts" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {candidate_text}")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_test_series() -> tuple[pd.Series, pd.Series, dict[str, Any], Path]:
    arrays_path = resolve_artifact_path("preprocessed_arrays.joblib")
    arrays = joblib.load(arrays_path)
    target_name = arrays.get("target_name", TARGET_FALLBACK)
    train_index = pd.to_datetime(arrays["train_index"])
    test_index = pd.to_datetime(arrays["test_index"])

    train = pd.Series(arrays["y_train"], index=train_index, name=target_name, dtype=np.float64)
    test = pd.Series(arrays["y_test"], index=test_index, name=target_name, dtype=np.float64)
    return train, test, arrays, arrays_path


def load_split_info() -> tuple[dict[str, Any], Path]:
    split_info_path = resolve_artifact_path("split_info.json")
    split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
    return split_info, split_info_path


def numeric_series(values: np.ndarray | pd.Series | list[float]) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=np.float64))


def compute_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    previous_actual: pd.Series,
) -> float:
    actual_direction = np.sign(actual.to_numpy(dtype=np.float64) - previous_actual.to_numpy(dtype=np.float64))
    predicted_direction = np.sign(predicted.to_numpy(dtype=np.float64) - previous_actual.to_numpy(dtype=np.float64))
    return float((actual_direction == predicted_direction).mean() * 100.0)


def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    previous_actual: pd.Series,
) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=np.float64)
    predicted_values = predicted.to_numpy(dtype=np.float64)
    errors = actual_values - predicted_values
    denominator = np.clip(np.abs(actual_values), MAPE_EPSILON, None)
    smape_denominator = np.abs(actual_values) + np.abs(predicted_values) + MAPE_EPSILON

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(np.mean(np.abs(errors) / denominator) * 100.0),
        "smape": float(np.mean(2.0 * np.abs(errors) / smape_denominator) * 100.0),
        "r2": float(r2_score(actual_values, predicted_values)),
        "directional_accuracy": compute_directional_accuracy(actual, predicted, previous_actual),
    }


def build_previous_actual_series(train: pd.Series, test: pd.Series) -> pd.Series:
    previous_values = np.concatenate(([train.iloc[-1]], test.iloc[:-1].to_numpy(dtype=np.float64)))
    return pd.Series(previous_values, index=test.index, name="previous_actual", dtype=np.float64)


def maybe_load_auto_arima() -> Any | None:
    try:
        from pmdarima.arima import auto_arima

        return auto_arima
    except Exception:
        return None


def forecast_naive(train: pd.Series, test: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    predictions = np.concatenate(([train.iloc[-1]], test.iloc[:-1].to_numpy(dtype=np.float64)))
    series = pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)
    config = {"strategy": "previous_actual"}
    return series, config


def forecast_moving_average(train: pd.Series, test: pd.Series, window: int = MOVING_AVERAGE_WINDOW) -> tuple[pd.Series, dict[str, Any]]:
    history = train.to_numpy(dtype=np.float64).tolist()
    predictions: list[float] = []

    for actual_value in test.to_numpy(dtype=np.float64):
        predictions.append(float(np.mean(history[-window:])))
        history.append(float(actual_value))

    series = pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)
    config = {"window": window, "strategy": "walk_forward_mean"}
    return series, config


def forecast_simple_exponential_smoothing(
    train: pd.Series,
    test: pd.Series,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fit = SimpleExpSmoothing(numeric_series(train)).fit(optimized=True)

    alpha = float(fit.params.get("smoothing_level", np.nan))
    final_level = float(np.asarray(fit.level)[-1])
    predictions: list[float] = []
    level = final_level

    for actual_value in test.to_numpy(dtype=np.float64):
        predictions.append(level)
        level = alpha * float(actual_value) + (1.0 - alpha) * level

    config = {
        "model_type": "SimpleExpSmoothing",
        "smoothing_level": alpha,
        "initial_level": float(fit.params.get("initial_level", np.nan)),
        "fitted_level_last": final_level,
    }
    search_frame = pd.DataFrame([config])
    series = pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)
    return series, config, search_frame


def fit_arima_model(
    train_values: np.ndarray,
    order: tuple[int, int, int],
    trend: str = "n",
    seasonal_order: tuple[int, int, int, int] | None = None,
):
    model_kwargs: dict[str, Any] = {
        "order": order,
        "trend": trend,
        "enforce_stationarity": False,
        "enforce_invertibility": False,
    }
    if seasonal_order is not None:
        model_kwargs["seasonal_order"] = seasonal_order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        return ARIMA(numeric_series(train_values), **model_kwargs).fit(method_kwargs={"maxiter": 200})


def select_arima_configuration(train: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    auto_arima = maybe_load_auto_arima()
    numeric_train = train.to_numpy(dtype=np.float64)

    if auto_arima is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                numeric_train,
                start_p=0,
                start_q=0,
                max_p=3,
                max_q=3,
                d=None,
                max_d=2,
                seasonal=False,
                stepwise=True,
                information_criterion="aic",
                suppress_warnings=True,
                error_action="ignore",
            )

        config = {
            "selection_method": "pmdarima.auto_arima",
            "order": tuple(int(value) for value in model.order),
            "trend": "n",
            "aic": float(model.aic()),
        }
        search_frame = pd.DataFrame(
            [
                {
                    "order": str(config["order"]),
                    "trend": config["trend"],
                    "aic": config["aic"],
                    "selection_method": config["selection_method"],
                    "status": "selected",
                }
            ]
        )
        return config, search_frame

    search_rows: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_aic = math.inf

    for p, d, q in product(range(4), range(3), range(4)):
        try:
            fit = fit_arima_model(numeric_train, order=(p, d, q), trend="n")
            aic = float(fit.aic)
            row = {
                "order": str((p, d, q)),
                "trend": "n",
                "aic": aic,
                "selection_method": "statsmodels_grid_search",
                "status": "ok",
            }
            search_rows.append(row)
            if math.isfinite(aic) and aic < best_aic:
                best_aic = aic
                best_config = {
                    "selection_method": "statsmodels_grid_search",
                    "order": (p, d, q),
                    "trend": "n",
                    "aic": aic,
                }
        except Exception as exc:
            search_rows.append(
                {
                    "order": str((p, d, q)),
                    "trend": "n",
                    "aic": None,
                    "selection_method": "statsmodels_grid_search",
                    "status": f"failed: {type(exc).__name__}",
                }
            )

    if best_config is None:
        raise RuntimeError("ARIMA model selection failed for all candidate orders.")

    search_frame = pd.DataFrame(search_rows).sort_values(["aic", "order"], na_position="last").reset_index(drop=True)
    return best_config, search_frame


def forecast_arima_walkforward(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    trend: str = "n",
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> pd.Series:
    history = train.to_numpy(dtype=np.float64)
    result = fit_arima_model(history, order=order, trend=trend, seasonal_order=seasonal_order)
    predictions: list[float] = []

    for actual_value in test.to_numpy(dtype=np.float64):
        next_forecast = result.forecast(steps=1)
        predictions.append(float(np.asarray(next_forecast)[0]))
        try:
            result = result.append([float(actual_value)], refit=False)
        except Exception:
            history = np.append(history, float(actual_value))
            result = fit_arima_model(history, order=order, trend=trend, seasonal_order=seasonal_order)

    return pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)


def select_sarima_configuration(train: pd.Series, seasonal_period: int = SEASONAL_PERIOD) -> tuple[dict[str, Any], pd.DataFrame]:
    auto_arima = maybe_load_auto_arima()
    numeric_train = train.to_numpy(dtype=np.float64)

    if auto_arima is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                numeric_train,
                start_p=0,
                start_q=0,
                max_p=2,
                max_q=2,
                d=None,
                max_d=2,
                start_P=0,
                start_Q=0,
                max_P=1,
                max_Q=1,
                D=None,
                max_D=1,
                m=seasonal_period,
                seasonal=True,
                stepwise=True,
                information_criterion="aic",
                suppress_warnings=True,
                error_action="ignore",
            )

        config = {
            "selection_method": "pmdarima.auto_arima",
            "order": tuple(int(value) for value in model.order),
            "seasonal_order": tuple(int(value) for value in model.seasonal_order),
            "trend": "n",
            "aic": float(model.aic()),
            "seasonal_period": seasonal_period,
        }
        search_frame = pd.DataFrame(
            [
                {
                    "order": str(config["order"]),
                    "seasonal_order": str(config["seasonal_order"]),
                    "trend": config["trend"],
                    "aic": config["aic"],
                    "selection_method": config["selection_method"],
                    "status": "selected",
                }
            ]
        )
        return config, search_frame

    search_rows: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_aic = math.inf

    for p, d, q in product(range(3), range(2), range(3)):
        for P, D, Q in product(range(2), range(2), range(2)):
            seasonal_order = (P, D, Q, seasonal_period)
            try:
                fit = fit_arima_model(numeric_train, order=(p, d, q), trend="n", seasonal_order=seasonal_order)
                aic = float(fit.aic)
                row = {
                    "order": str((p, d, q)),
                    "seasonal_order": str(seasonal_order),
                    "trend": "n",
                    "aic": aic,
                    "selection_method": "statsmodels_grid_search",
                    "status": "ok",
                }
                search_rows.append(row)
                if math.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_config = {
                        "selection_method": "statsmodels_grid_search",
                        "order": (p, d, q),
                        "seasonal_order": seasonal_order,
                        "trend": "n",
                        "aic": aic,
                        "seasonal_period": seasonal_period,
                    }
            except Exception as exc:
                search_rows.append(
                    {
                        "order": str((p, d, q)),
                        "seasonal_order": str(seasonal_order),
                        "trend": "n",
                        "aic": None,
                        "selection_method": "statsmodels_grid_search",
                        "status": f"failed: {type(exc).__name__}",
                    }
                )

    if best_config is None:
        raise RuntimeError("SARIMA model selection failed for all candidate orders.")

    search_frame = pd.DataFrame(search_rows).sort_values(["aic", "order", "seasonal_order"], na_position="last")
    search_frame = search_frame.reset_index(drop=True)
    return best_config, search_frame


def save_predictions_csv(model_result: ModelResult, actual: pd.Series, previous_actual: pd.Series) -> None:
    frame = pd.DataFrame(
        {
            "date": actual.index,
            "actual": actual.to_numpy(dtype=np.float64),
            "prediction": model_result.predictions.to_numpy(dtype=np.float64),
            "previous_actual": previous_actual.to_numpy(dtype=np.float64),
        }
    )
    frame["error"] = frame["actual"] - frame["prediction"]
    frame["absolute_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] ** 2
    frame["ape_pct"] = np.abs(frame["error"]) / np.clip(np.abs(frame["actual"]), MAPE_EPSILON, None) * 100.0

    legacy_output_path = RESULTS_DIR / f"{model_result.slug}_predictions.csv"
    requested_output_path = PREDICTIONS_DIR / f"{model_result.slug}_predictions.csv"
    frame.to_csv(legacy_output_path, index=False)
    frame.to_csv(requested_output_path, index=False)


def render_forecast_plot(train: pd.Series, actual: pd.Series, model_result: ModelResult) -> None:
    context = pd.concat([train.tail(PLOT_CONTEXT_POINTS), actual])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(context.index, context.to_numpy(dtype=np.float64), color="black", linewidth=1.5, label="Actual")
    ax.plot(
        model_result.predictions.index,
        model_result.predictions.to_numpy(dtype=np.float64),
        color="#1f77b4",
        linewidth=1.5,
        linestyle="--",
        label=f"{model_result.model_name} forecast",
    )
    ax.axvline(actual.index[0], color="gray", linestyle=":", linewidth=1, label="Test start")
    ax.set_title(f"{model_result.model_name}: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel(actual.name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_result.slug}_forecast.png", dpi=180)
    fig.savefig(RESULTS_DIR / f"{model_result.slug}_forecast.png", dpi=180)
    plt.close(fig)


def render_comparison_plot(actual: pd.Series, model_results: list[ModelResult]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual.index, actual.to_numpy(dtype=np.float64), color="black", linewidth=1.8, label="Actual", zorder=5)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for color, result in zip(colors, model_results):
        ax.plot(
            result.predictions.index,
            result.predictions.to_numpy(dtype=np.float64),
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=result.model_name,
        )
    ax.set_title("Statistical Baselines: Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel(actual.name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "statistical_model_comparison.png", dpi=180)
    fig.savefig(RESULTS_DIR / "statistical_model_comparison.png", dpi=180)
    plt.close(fig)


def save_search_artifacts(
    moving_average_summary: pd.DataFrame,
    ets_summary: pd.DataFrame,
    arima_search: pd.DataFrame,
    sarima_search: pd.DataFrame,
) -> None:
    moving_average_summary.to_csv(RESULTS_DIR / "moving_average_window_search.csv", index=False)
    ets_summary.to_csv(RESULTS_DIR / "ets_model_search.csv", index=False)
    arima_search.to_csv(RESULTS_DIR / "arima_model_search.csv", index=False)
    sarima_search.to_csv(RESULTS_DIR / "sarima_model_search.csv", index=False)


def save_model_selection_summary(
    train: pd.Series,
    test: pd.Series,
    arrays_path: Path,
    split_info_path: Path,
    ets_config: dict[str, Any],
    arima_config: dict[str, Any],
    sarima_config: dict[str, Any],
) -> None:
    summary = {
        "data_source": str(arrays_path),
        "split_info_source": str(split_info_path),
        "train_start": train.index.min().date().isoformat(),
        "train_end": train.index.max().date().isoformat(),
        "test_start": test.index.min().date().isoformat(),
        "test_end": test.index.max().date().isoformat(),
        "moving_average_window": MOVING_AVERAGE_WINDOW,
        "ets": ets_config,
        "arima": arima_config,
        "sarima": sarima_config,
    }
    (RESULTS_DIR / "model_selection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def build_leaderboard_frame(model_results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in model_results:
        rows.append(
            {
                "model": result.model_name,
                "slug": result.slug,
                "mae": result.metrics["mae"],
                "rmse": result.metrics["rmse"],
                "mape": result.metrics["mape"],
                "smape": result.metrics["smape"],
                "r2": result.metrics["r2"],
                "directional_accuracy": result.metrics["directional_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "mae", "mape"], ascending=True).reset_index(drop=True)


def save_metrics_tables(model_results: list[ModelResult]) -> pd.DataFrame:
    leaderboard = build_leaderboard_frame(model_results)
    leaderboard.to_csv(RESULTS_DIR / "statistical_leaderboard.csv", index=False)

    detailed_rows = []
    for result in model_results:
        detailed_rows.append(
            {
                "model_family": result.model_family,
                "model_name": result.model_name,
                "slug": result.slug,
                "mae": result.metrics["mae"],
                "rmse": result.metrics["rmse"],
                "mape": result.metrics["mape"],
                "smape": result.metrics["smape"],
                "r2": result.metrics["r2"],
                "directional_accuracy": result.metrics["directional_accuracy"],
                "config": json.dumps(result.config, default=str, sort_keys=True),
                "notes": result.notes,
            }
        )

    detailed_frame = pd.DataFrame(detailed_rows).sort_values(["rmse", "mae", "mape"], ascending=True)
    detailed_frame.to_csv(RESULTS_DIR / "statistical_model_metrics.csv", index=False)
    return leaderboard


def upsert_metrics_registry(model_results: list[ModelResult]) -> pd.DataFrame:
    new_rows = pd.DataFrame(
        [
            {
                "model_family": result.model_family,
                "model_name": result.model_name,
                "MAE": result.metrics["mae"],
                "RMSE": result.metrics["rmse"],
                "MAPE": result.metrics["mape"],
                "R2": result.metrics["r2"],
            }
            for result in model_results
        ]
    )

    if METRICS_REGISTRY_PATH.exists():
        existing = pd.read_csv(METRICS_REGISTRY_PATH)
        for column in new_rows.columns:
            if column not in existing.columns:
                existing[column] = np.nan
        if "model_family" in existing.columns:
            existing = existing[existing["model_family"] != "statistical"]
        registry = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        registry = new_rows

    registry = registry[["model_family", "model_name", "MAE", "RMSE", "MAPE", "R2"]]
    registry = registry.sort_values(["model_family", "RMSE", "MAE", "MAPE"], ascending=True).reset_index(drop=True)
    registry.to_csv(METRICS_REGISTRY_PATH, index=False)
    return registry


def format_metric(value: float, decimals: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def build_metrics_markdown_table(leaderboard: pd.DataFrame) -> str:
    lines = [
        "| Model | MAE | RMSE | MAPE (%) | R^2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in leaderboard.itertuples():
        lines.append(
            f"| {row.model} | {format_metric(row.mae)} | {format_metric(row.rmse)} | "
            f"{format_metric(row.mape)} | {format_metric(row.r2)} |"
        )
    return "\n".join(lines)


def upsert_research_log_section(section_markdown: str) -> None:
    existing = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    header_index = existing.find(LOG_SECTION_HEADER)
    section_markdown = section_markdown.strip()

    if header_index == -1:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n"
        updated += section_markdown + "\n"
    else:
        next_header_index = existing.find("\n## ", header_index + len(LOG_SECTION_HEADER))
        prefix = existing[:header_index].rstrip()
        suffix = existing[next_header_index:].lstrip("\n") if next_header_index != -1 else ""
        updated = prefix
        if updated:
            updated += "\n\n"
        updated += section_markdown + "\n"
        if suffix:
            updated += "\n" + suffix

    RESEARCH_LOG_PATH.write_text(updated, encoding="utf-8")


def append_statistical_research_log(
    leaderboard: pd.DataFrame,
    split_info: dict[str, Any],
    arrays_path: Path,
    split_info_path: Path,
    ets_config: dict[str, Any],
    arima_config: dict[str, Any],
    sarima_config: dict[str, Any],
) -> None:
    best_row = leaderboard.iloc[0]
    metrics_table = build_metrics_markdown_table(leaderboard)

    moving_average_mae = float(leaderboard.loc[leaderboard["slug"] == "moving_average_7d", "mae"].iloc[0])
    moving_average_rmse = float(leaderboard.loc[leaderboard["slug"] == "moving_average_7d", "rmse"].iloc[0])
    naive_rmse = float(leaderboard.loc[leaderboard["slug"] == "naive", "rmse"].iloc[0])

    findings = [
        f"- Best baseline by RMSE: `{best_row.model}` (RMSE={format_metric(best_row.rmse)}, MAE={format_metric(best_row.mae)}, R^2={format_metric(best_row.r2)})",
        f"- The fixed 7-day moving average scored MAE={format_metric(moving_average_mae)} and was {'better' if moving_average_rmse < naive_rmse else 'worse'} than naive persistence on RMSE.",
        f"- `SimpleExpSmoothing` optimized to alpha={format_metric(float(ets_config['smoothing_level']))}, then updated recursively over the test horizon.",
        f"- SARIMA enforced the required seasonal period of {SEASONAL_PERIOD}; the selected seasonal order was {tuple(sarima_config['seasonal_order'])}.",
    ]

    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Evaluation Setup",
        f"- Train target loaded from `{arrays_path.relative_to(PROJECT_ROOT).as_posix()}` and split metadata from `{split_info_path.relative_to(PROJECT_ROOT).as_posix()}`.",
        f"- Target column: `{split_info.get('target_column', TARGET_FALLBACK)}`",
        f"- Train window: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_size']} rows)",
        f"- Test window: {split_info['test_start']} to {split_info['test_end']} ({split_info['test_size']} rows)",
        f"- Baselines evaluated: Naive persistence, 7-day moving average, `SimpleExpSmoothing`, ARIMA, and SARIMA(m={SEASONAL_PERIOD}).",
        "",
        "### Selected Configurations",
        "- Naive: previous observed actual value",
        f"- 7-day moving average: rolling mean over the latest {MOVING_AVERAGE_WINDOW} raw target observations",
        f"- ETS/SES: smoothing_level={format_metric(float(ets_config['smoothing_level']))}, fitted_level_last={format_metric(float(ets_config['fitted_level_last']))}",
        f"- ARIMA: order={tuple(arima_config['order'])}, selection={arima_config['selection_method']}, AIC={format_metric(float(arima_config['aic']))}",
        f"- SARIMA: order={tuple(sarima_config['order'])}, seasonal_order={tuple(sarima_config['seasonal_order'])}, selection={sarima_config['selection_method']}, AIC={format_metric(float(sarima_config['aic']))}",
        "",
        "### Test Metrics",
        metrics_table,
        "",
        "### Findings",
        *findings,
        "",
        "### Saved Artifacts",
        "- `results/metrics_registry.csv`",
        "- `results/predictions/statistical/*.csv`",
        "- `plots/statistical/*.png`",
        "- `results/statistical/statistical_leaderboard.csv`",
        "- `results/statistical/statistical_model_metrics.csv`",
        "- `results/statistical/model_selection_summary.json`",
    ]
    upsert_research_log_section("\n".join(section_lines))


def run_all_statistical_models() -> pd.DataFrame:
    ensure_dirs()
    train, test, _, arrays_path = load_train_test_series()
    split_info, split_info_path = load_split_info()
    previous_actual = build_previous_actual_series(train, test)

    naive_predictions, naive_config = forecast_naive(train, test)
    moving_average_predictions, moving_average_config = forecast_moving_average(train, test, MOVING_AVERAGE_WINDOW)
    ets_predictions, ets_config, ets_search = forecast_simple_exponential_smoothing(train, test)
    arima_config, arima_search = select_arima_configuration(train)
    arima_predictions = forecast_arima_walkforward(train, test, order=tuple(arima_config["order"]), trend=arima_config["trend"])
    sarima_config, sarima_search = select_sarima_configuration(train, seasonal_period=SEASONAL_PERIOD)
    sarima_predictions = forecast_arima_walkforward(
        train,
        test,
        order=tuple(sarima_config["order"]),
        trend=sarima_config["trend"],
        seasonal_order=tuple(sarima_config["seasonal_order"]),
    )

    model_specs = [
        (
            "Naive/Persistence",
            "naive",
            naive_predictions,
            naive_config,
            "Walk-forward persistence forecast using the last observed actual value.",
        ),
        (
            "Moving Average (7-day)",
            "moving_average_7d",
            moving_average_predictions,
            moving_average_config,
            "Walk-forward 7-day moving-average forecast on the raw target series.",
        ),
        (
            "ETS (SimpleExpSmoothing)",
            "ets",
            ets_predictions,
            ets_config,
            "Simple exponential smoothing with alpha optimized on the train split, then updated recursively on test observations.",
        ),
        (
            f"ARIMA{tuple(arima_config['order'])}",
            "arima",
            arima_predictions,
            arima_config,
            "ARIMA baseline selected by AIC and evaluated with one-step-ahead walk-forward forecasting.",
        ),
        (
            f"SARIMA{tuple(sarima_config['order'])}x{tuple(sarima_config['seasonal_order'])}",
            "sarima",
            sarima_predictions,
            sarima_config,
            "SARIMA baseline with seasonal period 21, selected by AIC and evaluated in walk-forward mode.",
        ),
    ]

    model_results: list[ModelResult] = []
    for model_name, slug, predictions, config, notes in model_specs:
        model_results.append(
            ModelResult(
                model_family="statistical",
                model_name=model_name,
                slug=slug,
                predictions=predictions,
                metrics=compute_metrics(test, predictions, previous_actual),
                config=config,
                notes=notes,
            )
        )

    if len(model_results) != 5:
        raise RuntimeError(f"Expected 5 statistical models, found {len(model_results)}.")

    moving_average_summary = pd.DataFrame(
        [{"window": MOVING_AVERAGE_WINDOW, "selection_method": "fixed_by_task_spec", "status": "selected"}]
    )
    save_search_artifacts(moving_average_summary, ets_search, arima_search, sarima_search)
    save_model_selection_summary(train, test, arrays_path, split_info_path, ets_config, arima_config, sarima_config)

    for model_result in model_results:
        save_predictions_csv(model_result, test, previous_actual)
        render_forecast_plot(train, test, model_result)

    render_comparison_plot(test, model_results)
    leaderboard = save_metrics_tables(model_results)
    registry = upsert_metrics_registry(model_results)
    append_statistical_research_log(
        leaderboard=leaderboard,
        split_info=split_info,
        arrays_path=arrays_path,
        split_info_path=split_info_path,
        ets_config=ets_config,
        arima_config=arima_config,
        sarima_config=sarima_config,
    )

    statistical_rows = registry[registry["model_family"] == "statistical"]
    if len(statistical_rows) != 5:
        raise RuntimeError(f"Expected 5 statistical rows in metrics_registry.csv, found {len(statistical_rows)}.")

    return leaderboard


def main() -> None:
    leaderboard = run_all_statistical_models()
    print(f"Saved statistical baselines to {RESULTS_DIR} and {PREDICTIONS_DIR} ({len(leaderboard)} models).")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "statistical"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"
EDA_SUMMARY_PATH = PROJECT_ROOT / "plots" / "eda" / "eda_summary.json"

LOG_SECTION_HEADER = "## Statistical Models"
TARGET_FALLBACK = "Market_Liquidity_Index"
SEASONALITY_THRESHOLD = 0.05
SEASONAL_PERIOD_FALLBACK = 21
TRAIN_SELECTION_WINDOWS = (7, 14, 30)
PLOT_CONTEXT_POINTS = 180
MAPE_EPSILON = 1e-6


@dataclass(frozen=True)
class ModelResult:
    name: str
    slug: str
    predictions: pd.Series
    metrics: dict[str, float]
    config: dict[str, Any]
    notes: str


def load_train_test_series() -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Load the exact target split saved by preprocessing."""
    arrays = joblib.load(ARTIFACTS_DIR / "preprocessed_arrays.joblib")
    train_index = pd.to_datetime(arrays["train_index"])
    test_index = pd.to_datetime(arrays["test_index"])
    target_name = arrays.get("target_name", TARGET_FALLBACK)

    train = pd.Series(arrays["y_train"], index=train_index, name=target_name, dtype=np.float64)
    test = pd.Series(arrays["y_test"], index=test_index, name=target_name, dtype=np.float64)
    return train, test, arrays


def load_split_info() -> dict[str, Any]:
    return json.loads((ARTIFACTS_DIR / "split_info.json").read_text(encoding="utf-8"))


def load_eda_summary() -> dict[str, Any]:
    return json.loads(EDA_SUMMARY_PATH.read_text(encoding="utf-8"))


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert to a simple numeric index to avoid irregular-date frequency issues."""
    return pd.Series(series.to_numpy(dtype=np.float64), dtype=np.float64)


def compute_directional_accuracy(actual: pd.Series, predicted: pd.Series, previous_actual: pd.Series) -> float:
    actual_direction = np.sign(actual.to_numpy(dtype=np.float64) - previous_actual.to_numpy(dtype=np.float64))
    predicted_direction = np.sign(predicted.to_numpy(dtype=np.float64) - previous_actual.to_numpy(dtype=np.float64))
    return float((actual_direction == predicted_direction).mean() * 100.0)


def compute_metrics(actual: pd.Series, predicted: pd.Series, previous_actual: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=np.float64)
    predicted_values = predicted.to_numpy(dtype=np.float64)
    errors = actual_values - predicted_values
    denominator = np.clip(np.abs(actual_values), MAPE_EPSILON, None)

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(np.mean(np.abs(errors) / denominator) * 100.0),
        "r2": float(r2_score(actual_values, predicted_values)),
        "directional_accuracy": compute_directional_accuracy(actual, predicted, previous_actual),
    }


def build_previous_actual_series(train: pd.Series, test: pd.Series) -> pd.Series:
    values = np.concatenate(([train.iloc[-1]], test.iloc[:-1].to_numpy(dtype=np.float64)))
    return pd.Series(values, index=test.index, name="previous_actual", dtype=np.float64)


def select_moving_average_window(train: pd.Series) -> tuple[int, list[dict[str, Any]]]:
    """Pick the window that minimizes one-step-ahead MAE on the train split."""
    selection_rows: list[dict[str, Any]] = []
    for window in TRAIN_SELECTION_WINDOWS:
        backtest_pred = train.rolling(window=window).mean().shift(1).dropna()
        backtest_actual = train.loc[backtest_pred.index]
        mae = float(np.mean(np.abs(backtest_actual.to_numpy(dtype=np.float64) - backtest_pred.to_numpy(dtype=np.float64))))
        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(backtest_actual.to_numpy(dtype=np.float64) - backtest_pred.to_numpy(dtype=np.float64))
                )
            )
        )
        selection_rows.append(
            {
                "window": window,
                "train_backtest_mae": mae,
                "train_backtest_rmse": rmse,
                "observations": int(len(backtest_pred)),
            }
        )

    best = min(selection_rows, key=lambda row: (row["train_backtest_mae"], row["train_backtest_rmse"], row["window"]))
    return int(best["window"]), selection_rows


def forecast_naive(train: pd.Series, test: pd.Series) -> pd.Series:
    values = np.concatenate(([train.iloc[-1]], test.iloc[:-1].to_numpy(dtype=np.float64)))
    return pd.Series(values, index=test.index, name="prediction", dtype=np.float64)


def forecast_moving_average(train: pd.Series, test: pd.Series, window: int) -> pd.Series:
    history = train.to_numpy(dtype=np.float64).tolist()
    predictions: list[float] = []

    for actual_value in test.to_numpy(dtype=np.float64):
        predictions.append(float(np.mean(history[-window:])))
        history.append(float(actual_value))

    return pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)


def build_ets_candidates(seasonality_detected: bool, seasonal_period: int, train_size: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {"trend": None, "damped_trend": False, "seasonal": None, "seasonal_periods": None},
        {"trend": "add", "damped_trend": False, "seasonal": None, "seasonal_periods": None},
        {"trend": "add", "damped_trend": True, "seasonal": None, "seasonal_periods": None},
    ]

    if seasonality_detected and train_size >= 2 * seasonal_period:
        candidates.extend(
            [
                {"trend": None, "damped_trend": False, "seasonal": "add", "seasonal_periods": seasonal_period},
                {"trend": "add", "damped_trend": False, "seasonal": "add", "seasonal_periods": seasonal_period},
                {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": seasonal_period},
            ]
        )

    return candidates


def select_ets_configuration(
    train: pd.Series,
    seasonality_detected: bool,
    seasonal_period: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    numeric_train = to_numeric_series(train)
    candidates = build_ets_candidates(seasonality_detected, seasonal_period, len(train))

    best_result: Any | None = None
    best_config: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model = ExponentialSmoothing(
                    numeric_train,
                    trend=candidate["trend"],
                    damped_trend=candidate["damped_trend"],
                    seasonal=candidate["seasonal"],
                    seasonal_periods=candidate["seasonal_periods"],
                    initialization_method="estimated",
                )
                fit_result = model.fit(optimized=True, use_brute=True)

            aic = float(getattr(fit_result, "aic", np.inf))
            row = {
                **candidate,
                "aic": aic,
                "status": "ok",
            }
            search_rows.append(row)
            if math.isfinite(aic) and (best_result is None or aic < best_result.aic):
                best_result = fit_result
                best_config = dict(candidate)
        except Exception as exc:
            search_rows.append({**candidate, "aic": None, "status": f"failed: {type(exc).__name__}"})

    if best_result is None or best_config is None:
        raise RuntimeError("ETS model search failed for all candidate configurations.")

    params = dict(best_result.params)
    params["aic"] = float(best_result.aic)
    return best_config, params, search_rows


def forecast_ets_walkforward(
    train: pd.Series,
    test: pd.Series,
    config: dict[str, Any],
    fitted_params: dict[str, Any],
) -> pd.Series:
    history = train.to_numpy(dtype=np.float64).tolist()
    predictions: list[float] = []

    initial_seasonal = fitted_params.get("initial_seasons")
    if isinstance(initial_seasonal, np.ndarray):
        initial_seasonal = initial_seasonal.tolist()
    if initial_seasonal is not None and len(initial_seasonal) == 0:
        initial_seasonal = None

    model_kwargs: dict[str, Any] = {
        "trend": config["trend"],
        "damped_trend": config["damped_trend"],
        "seasonal": config["seasonal"],
        "seasonal_periods": config["seasonal_periods"],
        "initialization_method": "known",
        "initial_level": fitted_params.get("initial_level"),
    }
    if config["trend"] is not None:
        model_kwargs["initial_trend"] = fitted_params.get("initial_trend")
    if config["seasonal"] is not None and initial_seasonal is not None:
        model_kwargs["initial_seasonal"] = initial_seasonal

    for actual_value in test.to_numpy(dtype=np.float64):
        numeric_history = pd.Series(history, dtype=np.float64)
        model = ExponentialSmoothing(numeric_history, **model_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit_result = model.fit(
                smoothing_level=fitted_params.get("smoothing_level"),
                smoothing_trend=fitted_params.get("smoothing_trend"),
                smoothing_seasonal=fitted_params.get("smoothing_seasonal"),
                damping_trend=fitted_params.get("damping_trend"),
                optimized=False,
            )
        predictions.append(float(fit_result.forecast(1).iloc[0]))
        history.append(float(actual_value))

    return pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)


def maybe_load_auto_arima() -> Any | None:
    try:
        from pmdarima.arima import auto_arima

        return auto_arima
    except Exception:
        return None


def select_arima_configuration(train: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    numeric_train = to_numeric_series(train)
    auto_arima = maybe_load_auto_arima()

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
                max_d=3,
                seasonal=False,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )

        order = tuple(int(value) for value in model.order)
        return (
            {
                "selection_method": "pmdarima.auto_arima",
                "order": order,
                "trend": "n",
                "aic": float(model.aic()),
            },
            [
                {
                    "order": str(order),
                    "trend": "n",
                    "aic": float(model.aic()),
                    "status": "selected",
                }
            ],
        )

    best_fit: Any | None = None
    best_config: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []

    for p, d, q in product(range(4), repeat=3):
        for trend in ("n", "c"):
            if d > 1 and trend == "c":
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    model = ARIMA(
                        numeric_train,
                        order=(p, d, q),
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit_result = model.fit(method_kwargs={"maxiter": 200})

                aic = float(fit_result.aic)
                row = {"order": str((p, d, q)), "trend": trend, "aic": aic, "status": "ok"}
                search_rows.append(row)
                if math.isfinite(aic) and (best_fit is None or aic < best_fit.aic):
                    best_fit = fit_result
                    best_config = {
                        "selection_method": "statsmodels_grid_search",
                        "order": (p, d, q),
                        "trend": trend,
                        "aic": aic,
                    }
            except Exception as exc:
                search_rows.append(
                    {
                        "order": str((p, d, q)),
                        "trend": trend,
                        "aic": None,
                        "status": f"failed: {type(exc).__name__}",
                    }
                )

    if best_fit is None or best_config is None:
        raise RuntimeError("ARIMA search failed for all candidate configurations.")

    return best_config, search_rows


def forecast_arima_walkforward(train: pd.Series, test: pd.Series, config: dict[str, Any]) -> pd.Series:
    numeric_train = to_numeric_series(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ARIMA(
            numeric_train,
            order=tuple(config["order"]),
            trend=config["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 200})

    predictions: list[float] = []
    for actual_value in test.to_numpy(dtype=np.float64):
        next_forecast = result.forecast(steps=1)
        predictions.append(float(next_forecast.iloc[0]))
        next_index = pd.RangeIndex(start=result.nobs, stop=result.nobs + 1)
        result = result.append(pd.Series([float(actual_value)], index=next_index), refit=False)

    return pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)


def select_sarima_configuration(train: pd.Series, seasonal_period: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    numeric_train = to_numeric_series(train)
    auto_arima = maybe_load_auto_arima()

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
                start_P=0,
                start_Q=0,
                max_P=1,
                max_Q=1,
                D=None,
                max_D=1,
                seasonal=True,
                m=seasonal_period,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )

        return (
            {
                "selection_method": "pmdarima.auto_arima",
                "order": tuple(int(value) for value in model.order),
                "seasonal_order": tuple(int(value) for value in model.seasonal_order),
                "trend": "n",
                "aic": float(model.aic()),
            },
            [
                {
                    "order": str(tuple(int(value) for value in model.order)),
                    "seasonal_order": str(tuple(int(value) for value in model.seasonal_order)),
                    "trend": "n",
                    "aic": float(model.aic()),
                    "status": "selected",
                }
            ],
        )

    best_fit: Any | None = None
    best_config: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []

    for p, d, q in product(range(4), repeat=3):
        for seasonal_order in product(range(2), repeat=3):
            P, D, Q = seasonal_order
            seasonal_tuple = (P, D, Q, seasonal_period)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    model = ARIMA(
                        numeric_train,
                        order=(p, d, q),
                        seasonal_order=seasonal_tuple,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit_result = model.fit(method_kwargs={"maxiter": 200})

                aic = float(fit_result.aic)
                row = {
                    "order": str((p, d, q)),
                    "seasonal_order": str(seasonal_tuple),
                    "trend": "n",
                    "aic": aic,
                    "status": "ok",
                }
                search_rows.append(row)
                if math.isfinite(aic) and (best_fit is None or aic < best_fit.aic):
                    best_fit = fit_result
                    best_config = {
                        "selection_method": "statsmodels_grid_search",
                        "order": (p, d, q),
                        "seasonal_order": seasonal_tuple,
                        "trend": "n",
                        "aic": aic,
                    }
            except Exception as exc:
                search_rows.append(
                    {
                        "order": str((p, d, q)),
                        "seasonal_order": str(seasonal_tuple),
                        "trend": "n",
                        "aic": None,
                        "status": f"failed: {type(exc).__name__}",
                    }
                )

    if best_fit is None or best_config is None:
        raise RuntimeError("SARIMA search failed for all candidate configurations.")

    return best_config, search_rows


def forecast_sarima_walkforward(train: pd.Series, test: pd.Series, config: dict[str, Any]) -> pd.Series:
    numeric_train = to_numeric_series(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = ARIMA(
            numeric_train,
            order=tuple(config["order"]),
            seasonal_order=tuple(config["seasonal_order"]),
            trend=config["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 200})

    predictions: list[float] = []
    for actual_value in test.to_numpy(dtype=np.float64):
        next_forecast = result.forecast(steps=1)
        predictions.append(float(next_forecast.iloc[0]))
        next_index = pd.RangeIndex(start=result.nobs, stop=result.nobs + 1)
        result = result.append(pd.Series([float(actual_value)], index=next_index), refit=False)

    return pd.Series(predictions, index=test.index, name="prediction", dtype=np.float64)


def save_predictions_csv(model_result: ModelResult, actual: pd.Series, previous_actual: pd.Series) -> None:
    frame = pd.DataFrame(
        {
            "DATE": actual.index,
            "actual": actual.to_numpy(dtype=np.float64),
            "prediction": model_result.predictions.to_numpy(dtype=np.float64),
            "previous_actual": previous_actual.to_numpy(dtype=np.float64),
        }
    )
    frame["error"] = frame["actual"] - frame["prediction"]
    frame["absolute_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] ** 2
    frame["ape_pct"] = np.abs(frame["error"]) / np.clip(np.abs(frame["actual"]), MAPE_EPSILON, None) * 100.0
    frame.to_csv(RESULTS_DIR / f"{model_result.slug}_predictions.csv", index=False)


def render_model_plot(train: pd.Series, actual: pd.Series, model_result: ModelResult) -> None:
    context = pd.concat([train.tail(PLOT_CONTEXT_POINTS), actual])
    plt.figure(figsize=(14, 6))
    plt.plot(context.index, context.to_numpy(dtype=np.float64), color="black", linewidth=1.5, label="Actual")
    plt.plot(
        model_result.predictions.index,
        model_result.predictions.to_numpy(dtype=np.float64),
        color="tab:blue",
        linewidth=1.5,
        label=f"{model_result.name} forecast",
    )
    plt.axvline(actual.index[0], color="gray", linestyle="--", linewidth=1, label="Test start")
    plt.title(f"{model_result.name}: forecast vs actual")
    plt.xlabel("Date")
    plt.ylabel(actual.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_result.slug}_forecast.png", dpi=180)
    plt.close()


def render_comparison_plot(actual: pd.Series, model_results: list[ModelResult]) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(actual.index, actual.to_numpy(dtype=np.float64), color="black", linewidth=1.7, label="Actual")
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for color, result in zip(color_cycle, model_results):
        plt.plot(
            result.predictions.index,
            result.predictions.to_numpy(dtype=np.float64),
            linewidth=1.2,
            color=color,
            alpha=0.85,
            label=result.name,
        )
    plt.title("Statistical model comparison on test split")
    plt.xlabel("Date")
    plt.ylabel(actual.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "statistical_model_comparison.png", dpi=180)
    plt.close()


def save_metrics_table(model_results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in model_results:
        rows.append(
            {
                "model": result.name,
                "slug": result.slug,
                "mae": result.metrics["mae"],
                "rmse": result.metrics["rmse"],
                "mape_pct": result.metrics["mape"],
                "r2": result.metrics["r2"],
                "directional_accuracy_pct": result.metrics["directional_accuracy"],
                "config": json.dumps(result.config, default=str, sort_keys=True),
                "notes": result.notes,
            }
        )

    metrics_frame = pd.DataFrame(rows).sort_values(["rmse", "mae", "mape_pct"], ascending=True).reset_index(drop=True)
    metrics_frame.to_csv(RESULTS_DIR / "statistical_model_metrics.csv", index=False)
    return metrics_frame


def save_search_artifacts(
    moving_average_rows: list[dict[str, Any]],
    ets_rows: list[dict[str, Any]],
    arima_rows: list[dict[str, Any]],
    sarima_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(moving_average_rows).to_csv(RESULTS_DIR / "moving_average_window_search.csv", index=False)
    pd.DataFrame(ets_rows).to_csv(RESULTS_DIR / "ets_model_search.csv", index=False)
    pd.DataFrame(arima_rows).to_csv(RESULTS_DIR / "arima_model_search.csv", index=False)
    if sarima_rows:
        pd.DataFrame(sarima_rows).to_csv(RESULTS_DIR / "sarima_model_search.csv", index=False)


def build_selection_summary(
    moving_average_window: int,
    moving_average_rows: list[dict[str, Any]],
    ets_config: dict[str, Any],
    ets_params: dict[str, Any],
    arima_config: dict[str, Any],
    sarima_config: dict[str, Any] | None,
    seasonality_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "moving_average": {
            "selected_window": moving_average_window,
            "candidates": moving_average_rows,
        },
        "ets": {
            "selected_configuration": ets_config,
            "aic": ets_params.get("aic"),
        },
        "arima": arima_config,
        "sarima": sarima_config,
        "seasonality": seasonality_info,
    }


def write_selection_summary(selection_summary: dict[str, Any]) -> None:
    (RESULTS_DIR / "model_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2, default=str),
        encoding="utf-8",
    )


def format_metric_value(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float, np.floating)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return "n/a"
        return f"{numeric_value:.{decimals}f}"
    return str(value)


def build_metrics_markdown_table(metrics_frame: pd.DataFrame) -> str:
    display_columns = [
        ("model", "Model"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("mape_pct", "MAPE (%)"),
        ("r2", "R^2"),
        ("directional_accuracy_pct", "Directional Accuracy (%)"),
    ]
    lines = [
        "| " + " | ".join(column_title for _, column_title in display_columns) + " |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    format_metric_value(row["mae"]),
                    format_metric_value(row["rmse"]),
                    format_metric_value(row["mape_pct"]),
                    format_metric_value(row["r2"]),
                    format_metric_value(row["directional_accuracy_pct"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def upsert_research_log_section(section_markdown: str) -> None:
    existing_content = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    section_markdown = section_markdown.strip()
    header_index = existing_content.find(LOG_SECTION_HEADER)

    if header_index == -1:
        updated_content = existing_content.rstrip()
        if updated_content:
            updated_content += "\n\n"
        updated_content += section_markdown + "\n"
    else:
        next_header_index = existing_content.find("\n## ", header_index + len(LOG_SECTION_HEADER))
        prefix = existing_content[:header_index].rstrip()
        suffix = ""
        if next_header_index != -1:
            suffix = existing_content[next_header_index:].lstrip("\n")

        updated_content = prefix
        if updated_content:
            updated_content += "\n\n"
        updated_content += section_markdown + "\n"
        if suffix:
            updated_content += "\n" + suffix

    RESEARCH_LOG_PATH.write_text(updated_content, encoding="utf-8")


def append_statistical_models_research_log(
    metrics_frame: pd.DataFrame,
    moving_average_window: int,
    ets_config: dict[str, Any],
    ets_params: dict[str, Any],
    arima_config: dict[str, Any],
    sarima_config: dict[str, Any],
    split_info: dict[str, Any],
    seasonality_info: dict[str, Any],
) -> None:
    best_row = metrics_frame.iloc[0]
    metrics_table = build_metrics_markdown_table(metrics_frame)

    seasonality_flag = "yes" if seasonality_info["seasonality_detected_for_ets"] else "no"
    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Evaluation Setup",
        f"- Train window: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_size']} rows)",
        f"- Test window: {split_info['test_start']} to {split_info['test_end']} ({split_info['test_size']} rows)",
        f"- Seasonal period from EDA: {seasonality_info['seasonal_period']} observations",
        f"- Seasonal strength from decomposition: {format_metric_value(seasonality_info['seasonal_strength'])} (ETS seasonal candidates enabled: {seasonality_flag})",
        "",
        "### Model Selection Summary",
        f"- Best overall model by RMSE: `{best_row['model']}` (RMSE={format_metric_value(best_row['rmse'])}, MAE={format_metric_value(best_row['mae'])}, MAPE={format_metric_value(best_row['mape_pct'])}%)",
        f"- Moving average selected window: {moving_average_window}",
        f"- ETS selected config: trend={ets_config['trend']}, damped_trend={ets_config['damped_trend']}, seasonal={ets_config['seasonal']}, seasonal_periods={ets_config['seasonal_periods']}, AIC={format_metric_value(ets_params.get('aic'))}",
        f"- ARIMA selected config: order={tuple(arima_config['order'])}, trend={arima_config['trend']}, AIC={format_metric_value(arima_config.get('aic'))}",
        f"- SARIMA selected config: order={tuple(sarima_config['order'])}, seasonal_order={tuple(sarima_config['seasonal_order'])}, trend={sarima_config['trend']}, AIC={format_metric_value(sarima_config.get('aic'))}",
        "",
        "### Test Metrics",
        metrics_table,
        "",
        "### Saved Artifacts",
        "- `results/statistical/statistical_model_metrics.csv`",
        "- `results/statistical/model_selection_summary.json`",
        "- `results/statistical/statistical_model_comparison.png`",
        "- `results/statistical/*_predictions.csv`",
        "- `results/statistical/*_forecast.png`",
        "- `results/statistical/*_search.csv`",
    ]
    upsert_research_log_section("\n".join(section_lines))


def run_all_statistical_models() -> pd.DataFrame:
    ensure_output_dirs()
    train, test, arrays = load_train_test_series()
    split_info = load_split_info()
    eda_summary = load_eda_summary()

    seasonal_period = int(eda_summary.get("decomposition", {}).get("period") or SEASONAL_PERIOD_FALLBACK)
    seasonal_strength = float(eda_summary.get("decomposition", {}).get("seasonal_strength", 0.0))
    seasonality_detected = seasonal_strength >= SEASONALITY_THRESHOLD
    seasonality_info = {
        "seasonal_period": seasonal_period,
        "seasonal_strength": seasonal_strength,
        "seasonality_detected_for_ets": seasonality_detected,
        "seasonality_threshold": SEASONALITY_THRESHOLD,
    }

    previous_actual = build_previous_actual_series(train, test)

    moving_average_window, moving_average_rows = select_moving_average_window(train)
    moving_average_predictions = forecast_moving_average(train, test, moving_average_window)

    ets_config, ets_params, ets_rows = select_ets_configuration(train, seasonality_detected, seasonal_period)
    ets_predictions = forecast_ets_walkforward(train, test, ets_config, ets_params)

    arima_config, arima_rows = select_arima_configuration(train)
    arima_predictions = forecast_arima_walkforward(train, test, arima_config)

    sarima_config, sarima_rows = select_sarima_configuration(train, seasonal_period)
    sarima_predictions = forecast_sarima_walkforward(train, test, sarima_config)

    model_results = [
        ModelResult(
            name="naive",
            slug="naive",
            predictions=forecast_naive(train, test),
            metrics={},
            config={"strategy": "previous_actual", "target_name": arrays.get("target_name", TARGET_FALLBACK)},
            notes="Walk-forward persistence baseline using the latest observed actual value.",
        ),
        ModelResult(
            name="moving_average",
            slug="moving_average",
            predictions=moving_average_predictions,
            metrics={},
            config={"window": moving_average_window, "selection_metric": "train_backtest_mae"},
            notes="Walk-forward moving average forecast using the selected historical window.",
        ),
        ModelResult(
            name="ets",
            slug="ets",
            predictions=ets_predictions,
            metrics={},
            config={**ets_config, "aic": ets_params.get("aic")},
            notes="Exponential smoothing walk-forward forecast using the lowest-AIC ETS configuration.",
        ),
        ModelResult(
            name="arima",
            slug="arima",
            predictions=arima_predictions,
            metrics={},
            config=arima_config,
            notes="ARIMA walk-forward forecast using the lowest-AIC autoregressive specification.",
        ),
        ModelResult(
            name="sarima",
            slug="sarima",
            predictions=sarima_predictions,
            metrics={},
            config={**sarima_config, "seasonal_period": seasonal_period},
            notes="Seasonal ARIMA walk-forward forecast evaluated with the 21-observation seasonal period from EDA.",
        ),
    ]

    for index, model_result in enumerate(model_results):
        metrics = compute_metrics(test, model_result.predictions, previous_actual)
        model_results[index] = ModelResult(
            name=model_result.name,
            slug=model_result.slug,
            predictions=model_result.predictions,
            metrics=metrics,
            config=model_result.config,
            notes=model_result.notes,
        )
        save_predictions_csv(model_results[index], test, previous_actual)
        render_model_plot(train, test, model_results[index])

    render_comparison_plot(test, model_results)
    metrics_frame = save_metrics_table(model_results)
    save_search_artifacts(moving_average_rows, ets_rows, arima_rows, sarima_rows)

    selection_summary = build_selection_summary(
        moving_average_window=moving_average_window,
        moving_average_rows=moving_average_rows,
        ets_config=ets_config,
        ets_params=ets_params,
        arima_config=arima_config,
        sarima_config=sarima_config,
        seasonality_info=seasonality_info,
    )
    write_selection_summary(selection_summary)
    append_statistical_models_research_log(
        metrics_frame=metrics_frame,
        moving_average_window=moving_average_window,
        ets_config=ets_config,
        ets_params=ets_params,
        arima_config=arima_config,
        sarima_config=sarima_config,
        split_info=split_info,
        seasonality_info=seasonality_info,
    )

    if len(metrics_frame) != 5:
        raise RuntimeError(
            f"Expected 5 statistical model rows in the metrics table, found {len(metrics_frame)}."
        )

    return metrics_frame


if __name__ == "__main__":
    metrics_frame = run_all_statistical_models()
    print(f"Saved statistical metrics to {RESULTS_DIR / 'statistical_model_metrics.csv'} ({len(metrics_frame)} rows).")

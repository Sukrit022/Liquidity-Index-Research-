from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


TARGET_COLUMN = "Market_Liquidity_Index"
DATE_COLUMN = "DATE"
DECOMPOSITION_PERIOD = 21
ACF_LAGS = 60
PACF_LAGS = 60


def resolve_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    workspace_root = project_root.parent

    dataset_candidates = [
        workspace_root / "market_liquidity_index.csv",
        project_root / "market_liquidity_index.csv",
        project_root / "Code" / "market_liquidity_index.csv",
    ]

    for candidate in dataset_candidates:
        if candidate.exists():
            dataset_path = candidate
            break
    else:
        formatted_candidates = "\n".join(str(path) for path in dataset_candidates)
        raise FileNotFoundError(
            "Unable to locate market_liquidity_index.csv. Checked:\n"
            f"{formatted_candidates}"
        )

    output_root = project_root
    plot_dir = output_root / "plots" / "eda"
    plot_dir.mkdir(parents=True, exist_ok=True)

    return {
        "script_path": script_path,
        "project_root": project_root,
        "workspace_root": workspace_root,
        "dataset_path": dataset_path,
        "plot_dir": plot_dir,
        "research_log_path": output_root / "RESEARCH_LOG.md",
        "summary_path": plot_dir / "eda_summary.json",
    }


def load_series(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    expected_columns = {DATE_COLUMN, TARGET_COLUMN}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if df[DATE_COLUMN].duplicated().any():
        duplicate_count = int(df[DATE_COLUMN].duplicated().sum())
        raise ValueError(f"Dataset contains {duplicate_count} duplicate date rows.")

    return df


def compute_stationarity(series: pd.Series) -> dict[str, dict[str, float | str]]:
    clean_series = series.dropna()

    adf_result = adfuller(clean_series, autolag="AIC")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_result = kpss(clean_series, regression="c", nlags="auto")

    kpss_note = ""
    for warning in caught_warnings:
        message = str(warning.message)
        if "smaller than the p-value returned" in message:
            kpss_note = "actual p-value is smaller than the reported 0.01 bound"
            break
        if "greater than the p-value returned" in message:
            kpss_note = "actual p-value is greater than the reported 0.10 bound"
            break

    return {
        "adf": {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "used_lags": int(adf_result[2]),
            "n_obs": int(adf_result[3]),
            "critical_values": {key: float(value) for key, value in adf_result[4].items()},
            "icbest": float(adf_result[5]),
            "decision": "stationary" if adf_result[1] < 0.05 else "non-stationary",
        },
        "kpss": {
            "statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "used_lags": int(kpss_result[2]),
            "critical_values": {key: float(value) for key, value in kpss_result[3].items()},
            "decision": "stationary" if kpss_result[1] >= 0.05 else "non-stationary",
            "note": kpss_note,
        },
    }


def linear_trend_r2(series: pd.Series) -> float:
    x = np.arange(len(series), dtype=float).reshape(-1, 1)
    y = series.to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(x, y)
    return float(model.score(x, y))


def save_time_series_plot(series: pd.Series, plot_dir: Path) -> None:
    rolling_window = 30
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series.index, series.values, color="#1f77b4", linewidth=1, label="Daily index")
    ax.plot(
        series.index,
        series.rolling(rolling_window).mean(),
        color="#d62728",
        linewidth=2,
        label=f"{rolling_window}-day rolling mean",
    )
    ax.set_title("Market Liquidity Index Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Liquidity Index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "liquidity_index_timeseries.png", dpi=200)
    plt.close(fig)


def save_distribution_plots(series: pd.Series, plot_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(series, kde=True, ax=axes[0, 0], color="#1f77b4", edgecolor=None)
    axes[0, 0].set_title("Distribution with KDE")
    axes[0, 0].set_xlabel("Liquidity Index")

    sns.boxplot(x=series, ax=axes[0, 1], color="#ff7f0e")
    axes[0, 1].set_title("Boxplot")
    axes[0, 1].set_xlabel("Liquidity Index")

    sns.violinplot(x=series, ax=axes[1, 0], color="#2ca02c")
    axes[1, 0].set_title("Violin Plot")
    axes[1, 0].set_xlabel("Liquidity Index")

    qqplot(series, line="s", ax=axes[1, 1], markerfacecolor="#9467bd", markeredgecolor="#9467bd")
    axes[1, 1].set_title("Q-Q Plot")

    fig.tight_layout()
    fig.savefig(plot_dir / "distribution_analysis.png", dpi=200)
    plt.close(fig)


def save_monthly_boxplot(series: pd.Series, plot_dir: Path) -> dict[int, float]:
    monthly_frame = series.to_frame(name=TARGET_COLUMN).copy()
    monthly_frame["month"] = monthly_frame.index.month
    monthly_frame["month_name"] = monthly_frame.index.strftime("%b")

    ordered_months = list(range(1, 13))
    month_labels = [pd.Timestamp(2000, month, 1).strftime("%b") for month in ordered_months]
    monthly_means = (
        monthly_frame.groupby("month")[TARGET_COLUMN].mean().reindex(ordered_months)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=monthly_frame,
        x="month_name",
        y=TARGET_COLUMN,
        order=month_labels,
        ax=ax,
        color="#17becf",
    )
    ax.set_title("Monthly Distribution of Liquidity Index")
    ax.set_xlabel("Month")
    ax.set_ylabel("Liquidity Index")
    fig.tight_layout()
    fig.savefig(plot_dir / "monthly_boxplot.png", dpi=200)
    plt.close(fig)

    return {int(index): float(value) for index, value in monthly_means.items()}


def save_acf_pacf_plots(series: pd.Series, plot_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plot_acf(series, lags=ACF_LAGS, ax=axes[0], zero=False)
    axes[0].set_title(f"ACF - Raw Series ({ACF_LAGS} Lags)")
    plot_pacf(series, lags=PACF_LAGS, ax=axes[1], zero=False, method="ywm")
    axes[1].set_title(f"PACF - Raw Series ({PACF_LAGS} Lags)")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_pacf_raw.png", dpi=200)
    plt.close(fig)

    diff_series = series.diff().dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plot_acf(diff_series, lags=ACF_LAGS, ax=axes[0], zero=False)
    axes[0].set_title(f"ACF - First Difference ({ACF_LAGS} Lags)")
    plot_pacf(diff_series, lags=PACF_LAGS, ax=axes[1], zero=False, method="ywm")
    axes[1].set_title(f"PACF - First Difference ({PACF_LAGS} Lags)")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_pacf_diff.png", dpi=200)
    plt.close(fig)


def save_seasonal_decomposition(series: pd.Series, plot_dir: Path) -> dict[str, float]:
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=DECOMPOSITION_PERIOD,
        extrapolate_trend="freq",
    )

    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    fig.tight_layout()
    fig.savefig(plot_dir / "seasonal_decomposition_21d.png", dpi=200)
    plt.close(fig)

    seasonal_strength = 1.0 - (
        np.nanvar(decomposition.resid) / np.nanvar(decomposition.seasonal + decomposition.resid)
    )
    trend_strength = 1.0 - (
        np.nanvar(decomposition.resid) / np.nanvar(decomposition.trend + decomposition.resid)
    )

    return {
        "seasonal_strength": float(np.clip(seasonal_strength, 0.0, 1.0)),
        "trend_strength": float(np.clip(trend_strength, 0.0, 1.0)),
    }


def longest_missing_streak(date_index: pd.DatetimeIndex) -> int:
    full_range = pd.date_range(date_index.min(), date_index.max(), freq="D")
    missing = ~full_range.isin(date_index)
    longest = current = 0

    for is_missing in missing:
        if is_missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


def build_summary(df: pd.DataFrame, dataset_path: Path, plot_dir: Path) -> dict[str, object]:
    series = df.set_index(DATE_COLUMN)[TARGET_COLUMN]
    diff_series = series.diff().dropna()

    weekday_counts = {
        str(key): int(value)
        for key, value in series.index.day_name().value_counts().sort_index().items()
    }
    descriptive = {key: float(value) for key, value in series.describe().to_dict().items()}
    percentiles = {
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }
    outlier_threshold = 3.0
    z_scores = (series - series.mean()) / series.std(ddof=0)
    outlier_count = int((z_scores.abs() > outlier_threshold).sum())

    seasonality_metrics = save_monthly_boxplot(series, plot_dir)
    decomposition_strengths = save_seasonal_decomposition(series, plot_dir)

    return {
        "dataset_path": str(dataset_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "date_range": {
            "start": str(series.index.min().date()),
            "end": str(series.index.max().date()),
            "calendar_span_days": int((series.index.max() - series.index.min()).days),
        },
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_dates": int(df[DATE_COLUMN].duplicated().sum()),
        "weekday_counts": weekday_counts,
        "longest_calendar_gap_days": longest_missing_streak(series.index),
        "descriptive_stats": descriptive,
        "percentiles": percentiles,
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "outliers_abs_z_gt_3": outlier_count,
        "stationarity": {
            "raw": compute_stationarity(series),
            "first_difference": compute_stationarity(diff_series),
        },
        "lag1_autocorrelation": float(series.autocorr(lag=1)),
        "lag5_autocorrelation": float(series.autocorr(lag=5)),
        "lag21_autocorrelation": float(series.autocorr(lag=21)),
        "linear_trend_r2": linear_trend_r2(series),
        "monthly_means": seasonality_metrics,
        "decomposition": {
            "period": DECOMPOSITION_PERIOD,
            **decomposition_strengths,
        },
    }


def format_stationarity_block(results: dict[str, dict[str, float | str]]) -> str:
    adf = results["adf"]
    kpss_result = results["kpss"]
    kpss_note = f" ({kpss_result['note']})" if kpss_result.get("note") else ""

    return (
        f"- ADF statistic: {adf['statistic']:.4f}, p-value: {adf['p_value']:.6f}, "
        f"lags: {adf['used_lags']}, decision: {adf['decision']}\n"
        f"- KPSS statistic: {kpss_result['statistic']:.4f}, p-value: {kpss_result['p_value']:.6f}, "
        f"lags: {kpss_result['used_lags']}, decision: {kpss_result['decision']}{kpss_note}"
    )


def build_research_notes(summary: dict[str, object]) -> list[str]:
    notes: list[str] = []
    raw_stationarity = summary["stationarity"]["raw"]
    diff_stationarity = summary["stationarity"]["first_difference"]
    decomposition = summary["decomposition"]

    if (
        raw_stationarity["adf"]["decision"] == "stationary"
        and raw_stationarity["kpss"]["decision"] == "stationary"
    ):
        notes.append(
            "The raw series is stationary by both ADF and KPSS, so models that assume a stable mean "
            "can be fit directly without mandatory differencing."
        )
    else:
        notes.append(
            "The raw series shows mixed evidence on stationarity, so differencing and scale-aware models "
            "should remain in the ablation set."
        )

    if (
        diff_stationarity["adf"]["decision"] == "stationary"
        and diff_stationarity["kpss"]["decision"] == "stationary"
    ):
        notes.append(
            "First differences are stationary by both tests, which supports ARIMA-style specifications "
            "with at most one order of differencing."
        )

    if summary["lag1_autocorrelation"] > 0.7:
        notes.append(
            "Strong lag-1 autocorrelation indicates persistence; naive, moving-average, ETS, and autoregressive "
            "baselines should be competitive."
        )
    elif summary["lag1_autocorrelation"] > 0.4:
        notes.append(
            "Moderate short-run autocorrelation suggests the series contains exploitable temporal structure "
            "for both statistical and sequence models."
        )
    else:
        notes.append(
            "Weak lag-1 autocorrelation implies limited persistence, so feature engineering and nonlinear models "
            "will matter more than simple carry-forward baselines."
        )

    if decomposition["trend_strength"] > 0.5:
        notes.append(
            "Trend strength is material, so trend-capable models such as ETS, ARIMA with drift, boosted trees "
            "on lagged features, and recurrent networks are justified."
        )
    else:
        notes.append(
            "Trend strength is limited, which lowers the value of heavy trend parameterization and increases the "
            "importance of local dynamics."
        )

    if decomposition["seasonal_strength"] > 0.3:
        notes.append(
            "A non-trivial 21-trading-day seasonal component is present. Seasonal lags, SARIMA candidates, and "
            "windowed DL models should include approximately monthly trading-cycle context."
        )
    else:
        notes.append(
            "The 21-trading-day seasonal component is weak, so monthly seasonality should be treated as optional "
            "rather than assumed."
        )

    if summary["outliers_abs_z_gt_3"] > 0:
        notes.append(
            "The distribution contains tail events beyond 3 standard deviations, which argues for robust error "
            "tracking and possibly scaled targets for deep-learning models."
        )

    if any(day in summary["weekday_counts"] for day in ["Saturday", "Sunday"]):
        weekend_points = summary["weekday_counts"].get("Saturday", 0) + summary["weekday_counts"].get(
            "Sunday", 0
        )
        if weekend_points > 0:
            notes.append(
                "A small number of weekend timestamps appear in the source data. Downstream models should treat "
                "the series as ordered trading observations rather than assume a perfectly regular business-day index."
            )

    return notes


def write_research_log(summary: dict[str, object], paths: dict[str, Path]) -> None:
    raw_stationarity = summary["stationarity"]["raw"]
    diff_stationarity = summary["stationarity"]["first_difference"]
    notes = build_research_notes(summary)

    monthly_means = summary["monthly_means"]
    strongest_month = max(monthly_means, key=monthly_means.get)
    weakest_month = min(monthly_means, key=monthly_means.get)
    strongest_month_label = pd.Timestamp(2000, int(strongest_month), 1).strftime("%B")
    weakest_month_label = pd.Timestamp(2000, int(weakest_month), 1).strftime("%B")

    log_text = f"""# Research Log

## EDA Initialization

### Dataset Provenance
- Source file used: `{summary['dataset_path']}`
- Output directory: `{paths['plot_dir']}`
- Generated by: `{paths['script_path']}`

### Dataset Summary
- Shape: {summary['shape']['rows']} rows x {summary['shape']['columns']} columns
- Columns: {', '.join(summary['columns'])}
- Date range: {summary['date_range']['start']} to {summary['date_range']['end']} ({summary['date_range']['calendar_span_days']} calendar days)
- Missing values: {summary['missing_values']}
- Duplicate dates: {summary['duplicate_dates']}
- Longest calendar gap in observations: {summary['longest_calendar_gap_days']} days
- Weekday counts: {summary['weekday_counts']}

### Key Statistics
- Mean: {summary['descriptive_stats']['mean']:.4f}
- Std. dev.: {summary['descriptive_stats']['std']:.4f}
- Min / Median / Max: {summary['descriptive_stats']['min']:.4f} / {summary['descriptive_stats']['50%']:.4f} / {summary['descriptive_stats']['max']:.4f}
- 1st / 99th percentile: {summary['percentiles']['p01']:.4f} / {summary['percentiles']['p99']:.4f}
- Skewness: {summary['skewness']:.4f}
- Kurtosis: {summary['kurtosis']:.4f}
- Observations with |z| > 3: {summary['outliers_abs_z_gt_3']}
- Lag autocorrelation (1, 5, 21): {summary['lag1_autocorrelation']:.4f}, {summary['lag5_autocorrelation']:.4f}, {summary['lag21_autocorrelation']:.4f}
- Linear trend R^2 over time index: {summary['linear_trend_r2']:.4f}

### Stationarity Tests
#### Raw Series
{format_stationarity_block(raw_stationarity)}

#### First-Differenced Series
{format_stationarity_block(diff_stationarity)}

### Trend and Seasonality
- Seasonal decomposition period: {summary['decomposition']['period']} trading observations
- Trend strength: {summary['decomposition']['trend_strength']:.4f}
- Seasonal strength: {summary['decomposition']['seasonal_strength']:.4f}
- Highest average month: {strongest_month_label} ({monthly_means[strongest_month]:.4f})
- Lowest average month: {weakest_month_label} ({monthly_means[weakest_month]:.4f})

### Distribution Analysis
- The series is left-skewed ({summary['skewness']:.4f}), indicating heavier downside excursions than upside spikes.
- Kurtosis is {summary['kurtosis']:.4f}, suggesting tails are close to but not exactly Gaussian.
- Tail observations are present and should be tracked carefully in MAE/RMSE comparisons.

### Saved EDA Artifacts
- `plots/eda/liquidity_index_timeseries.png`
- `plots/eda/distribution_analysis.png`
- `plots/eda/monthly_boxplot.png`
- `plots/eda/acf_pacf_raw.png`
- `plots/eda/acf_pacf_diff.png`
- `plots/eda/seasonal_decomposition_21d.png`
- `plots/eda/eda_summary.json`

### Researcher Notes
"""

    for note in notes:
        log_text += f"- {note}\n"

    paths["research_log_path"].write_text(log_text, encoding="utf-8")


def save_summary(summary: dict[str, object], summary_path: Path) -> None:
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    paths = resolve_paths()
    df = load_series(paths["dataset_path"])
    series = df.set_index(DATE_COLUMN)[TARGET_COLUMN]

    save_time_series_plot(series, paths["plot_dir"])
    save_distribution_plots(series, paths["plot_dir"])
    save_acf_pacf_plots(series, paths["plot_dir"])

    summary = build_summary(df, paths["dataset_path"], paths["plot_dir"])
    save_summary(summary, paths["summary_path"])
    write_research_log(summary, paths)

    print(f"Dataset used: {paths['dataset_path']}")
    print(f"Research log written to: {paths['research_log_path']}")
    print(f"Plots written to: {paths['plot_dir']}")


if __name__ == "__main__":
    main()

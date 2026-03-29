from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
FINAL_REPORT_PATH = PROJECT_ROOT / "FINAL_REPORT.md"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

SPLIT_INFO_PATH = ARTIFACTS_DIR / "split_info.json"
PREPROCESSED_ARRAYS_PATH = ARTIFACTS_DIR / "preprocessed_arrays.joblib"
OFFICIAL_LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.csv"
OFFICIAL_REGISTRY_PATH = RESULTS_DIR / "metrics_registry.csv"
ARCHIVED_MASTER_PATH = RESULTS_DIR / "MASTER_LEADERBOARD.csv"

LOG_SECTION_HEADER = "## Study Complete"
OFFICIAL_FAMILY_ORDER = ["ensemble", "ml", "statistical", "dl_advanced", "dl"]
ARCHIVED_FAMILY_ORDER = [
    "Statistical Baselines",
    "Classical ML",
    "RNN (LSTM/GRU)",
    "Advanced DL",
    "Ensemble/Hybrid",
]
MODEL_FAMILY_LABELS = {
    "ensemble": "Ensemble",
    "ml": "ML",
    "statistical": "Statistical",
    "dl_advanced": "Advanced DL",
    "dl": "Deep Learning RNN",
}
PLOT_REFERENCES = {
    "leaderboard": "./plots/leaderboard_comparison.png",
    "statistical": "./plots/statistical/statistical_model_comparison.png",
    "ml": "./plots/ml/rmse_comparison.png",
    "rnn_loss": "./plots/dl/lstm_loss_curve.png",
    "advanced": "./plots/dl_advanced/cnn_lstm_forecast.png",
    "ensemble": "./plots/ensemble/ensemble_comparison.png",
}


@dataclass(frozen=True)
class AppendixEntry:
    family: str
    model_name: str
    config_summary: str


APPENDIX_ENTRIES = [
    AppendixEntry(
        family="Statistical Baselines",
        model_name="arima",
        config_summary="ARIMA(order=(1,1,1), trend='n'); selected by auto_arima with train AIC=-1356.17.",
    ),
    AppendixEntry(
        family="Statistical Baselines",
        model_name="sarima",
        config_summary="SARIMA(order=(1,1,1), seasonal_order=(0,0,0,21), trend='n'); seasonal period=21.",
    ),
    AppendixEntry(
        family="Statistical Baselines",
        model_name="ets",
        config_summary="SimpleExpSmoothing; alpha=0.3724; initial_level=-1.4243; recursive walk-forward updates.",
    ),
    AppendixEntry(
        family="Statistical Baselines",
        model_name="naive",
        config_summary="Persistence baseline; predicts the previous observed actual value.",
    ),
    AppendixEntry(
        family="Statistical Baselines",
        model_name="moving_average",
        config_summary="Walk-forward moving average with window=7.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="SVR (Linear kernel)",
        config_summary="SVR with linear kernel; archive name omits C/epsilon, later approved run used C=1.0.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="ElasticNet (a=0.01, l1=0.5)",
        config_summary="ElasticNet(alpha=0.01, l1_ratio=0.5) on standardized engineered features.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Ridge Regression (a=10.0)",
        config_summary="Ridge(alpha=10.0).",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Ridge Regression (a=1.0)",
        config_summary="Ridge(alpha=1.0).",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Linear Regression",
        config_summary="Ordinary least squares regression on the 17 standardized features.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Lasso Regression (a=0.01)",
        config_summary="Lasso(alpha=0.01); later approved run used max_iter=10000.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Random Forest (n=200)",
        config_summary="Random forest with n_estimators=200; full tree-depth kwargs were not preserved in archive metadata.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Gradient Boosting (n=200)",
        config_summary="Gradient boosting regressor with n_estimators=200; remaining booster kwargs not persisted.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="LightGBM (n=300)",
        config_summary="LightGBM regressor with n_estimators=300; additional leaf and depth settings not retained.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="Extra Trees (n=200)",
        config_summary="ExtraTrees regressor with n_estimators=200; remaining forest kwargs not retained.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="XGBoost (n=300)",
        config_summary="XGBoost regressor with n_estimators=300; archive metadata does not retain learning-rate or depth.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="SVR (RBF kernel)",
        config_summary="SVR with RBF kernel; later approved run used C=10 and gamma='scale'.",
    ),
    AppendixEntry(
        family="Classical ML",
        model_name="KNN (k=5)",
        config_summary="KNeighborsRegressor with n_neighbors=5.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="GRU (vanilla, 64u)",
        config_summary="Single-layer GRU with 64 units.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="LSTM (vanilla, 64u)",
        config_summary="Single-layer LSTM with 64 units.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="LSTM (dropout=0.2)",
        config_summary="Single-layer LSTM with dropout=0.2.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="Stacked GRU (2-layer)",
        config_summary="Two-layer stacked GRU.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="Bidirectional GRU",
        config_summary="Bidirectional GRU encoder.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="Bidirectional LSTM",
        config_summary="Bidirectional LSTM encoder.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="Stacked LSTM (2-layer)",
        config_summary="Two-layer stacked LSTM.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="Deep LSTM (3-layer + BN)",
        config_summary="Three-layer LSTM stack with batch normalization.",
    ),
    AppendixEntry(
        family="RNN (LSTM/GRU)",
        model_name="LSTM (dropout=0.4)",
        config_summary="Single-layer LSTM with dropout=0.4.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="Transformer (4-head)",
        config_summary="Temporal transformer with 4 attention heads and sinusoidal positional encoding; archived note indicates 150 epochs.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="WaveNet Dilated CNN",
        config_summary="WaveNet-style dilated causal CNN with gated skip connections; exact dilation schedule not retained.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="CNN-GRU",
        config_summary="Conv1D encoder followed by a GRU decoder.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="Attention BiLSTM",
        config_summary="Bidirectional LSTM with attention pooling.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="Attention LSTM",
        config_summary="LSTM with attention pooling.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="TCN (Temporal Conv Net)",
        config_summary="Temporal convolutional network; exact residual-block depth was not persisted.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="CNN-LSTM",
        config_summary="Conv1D encoder followed by an LSTM decoder.",
    ),
    AppendixEntry(
        family="Advanced DL",
        model_name="CNN + Transformer",
        config_summary="Hybrid CNN encoder plus transformer block.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Top-5 Average",
        config_summary="Uniform average over the top 5 archived models by ranking.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Top-3 Average",
        config_summary="Uniform average over the top 3 archived models by ranking.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Weighted Average (inv-MAE)",
        config_summary="Weighted average with inverse-MAE weights.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Simple Average (all models)",
        config_summary="Uniform average across the archived model pool.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Stacking (Ridge meta)",
        config_summary="Stacking ensemble with a Ridge meta-learner.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Stacking (Linear meta)",
        config_summary="Stacking ensemble with a LinearRegression meta-learner.",
    ),
    AppendixEntry(
        family="Ensemble/Hybrid",
        model_name="Stacking (XGBoost meta)",
        config_summary="Stacking ensemble with an XGBoost meta-learner.",
    ),
]


def load_split_info() -> dict[str, object]:
    return json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))


def load_official_leaderboard() -> pd.DataFrame:
    leaderboard = pd.read_csv(OFFICIAL_LEADERBOARD_PATH)
    registry = pd.read_csv(OFFICIAL_REGISTRY_PATH)
    if len(leaderboard) != len(registry):
        raise ValueError(
            "results/leaderboard.csv and results/metrics_registry.csv should describe the same approved study rows."
        )
    return leaderboard.sort_values("rank").reset_index(drop=True)


def load_archived_master_leaderboard() -> pd.DataFrame:
    leaderboard = pd.read_csv(ARCHIVED_MASTER_PATH).copy()
    leaderboard.insert(0, "archived_rank", np.arange(1, len(leaderboard) + 1, dtype=np.int64))
    return leaderboard


def shorten_feature_name(feature_name: str) -> str:
    prefix = "Market_Liquidity_Index_"
    if feature_name.startswith(prefix):
        return feature_name[len(prefix) :]
    return feature_name


def compute_ridge_feature_importance(top_n: int = 8) -> pd.DataFrame:
    arrays = joblib.load(PREPROCESSED_ARRAYS_PATH)
    model = Ridge(alpha=1.0)
    model.fit(arrays["X_train_ml_scaled"], arrays["y_train_ml_scaled"])

    importance = pd.DataFrame(
        {
            "feature": [shorten_feature_name(name) for name in arrays["feature_names"]],
            "abs_coef": np.abs(np.asarray(model.coef_, dtype=np.float64).reshape(-1)),
        }
    )
    return importance.sort_values("abs_coef", ascending=False).head(top_n).reset_index(drop=True)


def build_family_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in OFFICIAL_FAMILY_ORDER:
        family_frame = leaderboard.loc[leaderboard["model_family"] == family].copy()
        if family_frame.empty:
            continue
        best_row = family_frame.sort_values(["RMSE", "MAE", "MAPE"], ascending=True).iloc[0]
        rows.append(
            {
                "Family": MODEL_FAMILY_LABELS.get(family, family),
                "Models": int(len(family_frame)),
                "Best Model": str(best_row["model_name"]),
                "Best RMSE": float(best_row["RMSE"]),
                "Mean RMSE": float(family_frame["RMSE"].mean()),
                "Best R2": float(family_frame["R2"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_improvement_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    naive_rmse = float(
        leaderboard.loc[leaderboard["model_name"] == "Naive/Persistence", "RMSE"].iloc[0]
    )
    arima_rmse = float(
        leaderboard.loc[leaderboard["model_name"] == "ARIMA(1, 1, 1)", "RMSE"].iloc[0]
    )

    rows: list[dict[str, object]] = []
    for row in leaderboard.head(3).itertuples(index=False):
        rmse = float(row.RMSE)
        rows.append(
            {
                "Model": row.model_name,
                "RMSE": rmse,
                "Improvement vs Naive (%)": (naive_rmse - rmse) / naive_rmse * 100.0,
                "Improvement vs ARIMA (%)": (arima_rmse - rmse) / arima_rmse * 100.0,
            }
        )

    top3_mean_rmse = float(leaderboard.head(3)["RMSE"].mean())
    rows.append(
        {
            "Model": "Top-3 mean RMSE",
            "RMSE": top3_mean_rmse,
            "Improvement vs Naive (%)": (naive_rmse - top3_mean_rmse) / naive_rmse * 100.0,
            "Improvement vs ARIMA (%)": (arima_rmse - top3_mean_rmse) / arima_rmse * 100.0,
        }
    )
    return pd.DataFrame(rows)


def format_official_leaderboard_table(leaderboard: pd.DataFrame) -> str:
    lines = [
        "| Rank | Family | Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R^2 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard.itertuples(index=False):
        lines.append(
            f"| {int(row.rank)} | {row.family_label} | {row.model_name} | "
            f"{float(row.RMSE):.4f} | {float(row.MAE):.4f} | {float(row.MAPE):.4f} | "
            f"{float(row.SMAPE):.4f} | {float(row.R2):.4f} |"
        )
    return "\n".join(lines)


def format_family_summary_table(family_summary: pd.DataFrame) -> str:
    lines = [
        "| Family | Models | Best Model | Best RMSE | Mean RMSE | Best R^2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in family_summary.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {int(row[1])} | {row[2]} | {float(row[3]):.4f} | "
            f"{float(row[4]):.4f} | {float(row[5]):.4f} |"
        )
    return "\n".join(lines)


def format_feature_table(feature_importance: pd.DataFrame) -> str:
    lines = [
        "| Feature | Absolute standardized coefficient |",
        "|---|---:|",
    ]
    for row in feature_importance.itertuples(index=False):
        lines.append(f"| {row.feature} | {float(row.abs_coef):.4f} |")
    return "\n".join(lines)


def format_improvement_table(improvement_summary: pd.DataFrame) -> str:
    lines = [
        "| Model | RMSE | Improvement vs Naive (%) | Improvement vs ARIMA (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in improvement_summary.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {float(row[1]):.6f} | {float(row[2]):.3f} | {float(row[3]):.3f} |"
        )
    return "\n".join(lines)


def format_appendix_tables(master_leaderboard: pd.DataFrame) -> str:
    entry_map = {(entry.family, entry.model_name): entry.config_summary for entry in APPENDIX_ENTRIES}
    sections: list[str] = []

    for family in ARCHIVED_FAMILY_ORDER:
        family_frame = master_leaderboard.loc[master_leaderboard["family"] == family].copy()
        if family_frame.empty:
            continue

        lines = [
            f"### {family}",
            "",
            "| Rank | Model | RMSE | Configuration summary |",
            "|---|---|---:|---|",
        ]
        for row in family_frame.itertuples(index=False):
            key = (family, row.model)
            if key not in entry_map:
                raise KeyError(f"Missing appendix configuration entry for {family} / {row.model}")
            lines.append(
                f"| {int(row.archived_rank)} | {row.model} | {float(row.rmse):.4f} | {entry_map[key]} |"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", text))


def build_final_report(
    split_info: dict[str, object],
    leaderboard: pd.DataFrame,
    family_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    master_leaderboard: pd.DataFrame,
) -> str:
    winner = leaderboard.iloc[0]
    runner_up = leaderboard.iloc[1]
    third_place = leaderboard.iloc[2]
    best_statistical = leaderboard.loc[leaderboard["model_family"] == "statistical"].iloc[0]
    best_ml = leaderboard.loc[leaderboard["model_family"] == "ml"].iloc[0]
    best_advanced_dl = leaderboard.loc[leaderboard["model_family"] == "dl_advanced"].iloc[0]
    best_rnn = leaderboard.loc[leaderboard["model_family"] == "dl"].iloc[0]
    naive = leaderboard.loc[leaderboard["model_name"] == "Naive/Persistence"].iloc[0]

    winner_vs_runner_up = float(runner_up["RMSE"]) - float(winner["RMSE"])
    runner_up_vs_third = float(third_place["RMSE"]) - float(runner_up["RMSE"])
    winner_vs_naive_pct = (float(naive["RMSE"]) - float(winner["RMSE"])) / float(naive["RMSE"]) * 100.0
    winner_vs_arima_pct = (
        float(best_statistical["RMSE"]) - float(winner["RMSE"])
    ) / float(best_statistical["RMSE"]) * 100.0

    report_lines = [
        "# Liquidity Index Ablation Study Final Report",
        "",
        f"Generated from `{OFFICIAL_LEADERBOARD_PATH.relative_to(PROJECT_ROOT).as_posix()}`, "
        f"`{OFFICIAL_REGISTRY_PATH.relative_to(PROJECT_ROOT).as_posix()}`, and "
        f"`{ARCHIVED_MASTER_PATH.relative_to(PROJECT_ROOT).as_posix()}`.",
        "",
        "## 1. Abstract",
        "",
        (
            "This report closes the liquidity index ablation study by comparing 22 approved models across "
            "statistical, classical machine learning, recurrent neural network, advanced deep learning, and "
            "ensemble families on a fixed chronological holdout. The best model on the approved leaderboard is "
            f"`{winner['model_name']}` with RMSE={float(winner['RMSE']):.4f}, MAE={float(winner['MAE']):.4f}, "
            f"MAPE={float(winner['MAPE']):.2f}%, and R^2={float(winner['R2']):.4f}."
        ),
        (
            f"The runner-up `{runner_up['model_name']}` trails by only {winner_vs_runner_up:.6f} RMSE, and the "
            f"third-place `{third_place['model_name']}` is only {runner_up_vs_third:.6f} behind the runner-up, so "
            "the practical difference among the top three models is negligible even though the ranking is stable."
        ),
        (
            "The dominant empirical pattern is that persistence-aware linear and statistical models beat both "
            "tree ensembles and deep neural architectures, implying that the liquidity index is largely driven by "
            "recent history and smooth local trend rather than high-order nonlinear structure."
        ),
        (
            "A broader 42-model archived appendix is retained for reproducibility, but all main conclusions in this "
            "document are drawn from the approved 22-row leaderboard produced at the end of step-8."
        ),
        "",
        "## 2. Dataset & Problem Statement",
        "",
        (
            "The target is the daily `Market_Liquidity_Index` series stored in `Code/market_liquidity_index.csv`. "
            f"The raw dataset contains {int(split_info['row_count_raw'])} rows, which became "
            f"{int(split_info['row_count_model_ready'])} model-ready observations after feature warmup. The train "
            f"window runs from {split_info['train_start']} through {split_info['train_end']} "
            f"({int(split_info['train_size'])} rows), and the test window runs from {split_info['test_start']} "
            f"through {split_info['test_end']} ({int(split_info['test_size'])} rows)."
        ),
        (
            "The forecasting task is one-step-ahead prediction of the liquidity index under a strict chronological "
            "split. This is a realistic market setting: the model sees only past observations, and all evaluation "
            "happens on a later period that includes post-2022 market conditions."
        ),
        (
            f"Missing values were handled with {split_info['missing_value_strategy']}. Outliers were treated with "
            f"{split_info['outlier_method']}, which capped {int(split_info['outlier_stats']['outliers_capped'])} "
            "extreme observations rather than dropping them, preserving rare stress regimes while limiting scale "
            "distortion for downstream learners."
        ),
        "",
        "## 3. Methodology",
        "",
        (
            "Feature engineering followed the approved preprocessing pipeline. Eight lag terms "
            f"({split_info['lag_windows']}), rolling means and rolling standard deviations over windows "
            f"{split_info['rolling_windows']}, and three calendar variables "
            f"({split_info['date_features']}) were built, giving {int(split_info['feature_count'])} features in total."
        ),
        (
            "Leakage prevention was handled conservatively. The split remained chronological, all scalers were fit on "
            "the training partition only, `StandardScaler` was reserved for classical ML models, and `MinMaxScaler` "
            "was reserved for the deep-learning families. Statistical baselines were evaluated in walk-forward mode, "
            "while the supervised ML and DL families used the frozen train/test split from `artifacts/preprocessed_arrays.joblib`."
        ),
        "The approved model families were:",
        "- Statistical baselines: Naive, 7-day Moving Average, ETS, ARIMA, and SARIMA.",
        "- Classical ML: LinearRegression, Ridge, Lasso, linear SVR, RBF SVR, RandomForest, XGBoost, and LightGBM.",
        "- Deep Learning RNN: LSTM, Bidirectional LSTM, and GRU on univariate lookback-30 sequences.",
        "- Advanced DL: CNN-LSTM, Attention LSTM, and Temporal Transformer on multivariate lookback-30 sequences.",
        "- Ensemble: weighted top-5 ensemble, top-3 ML simple average, and linear stacking.",
        (
            "All families were scored with MAE, RMSE, MAPE, R^2, and SMAPE. For the final ranking, RMSE is the "
            "primary criterion because it penalizes large forecast misses more strongly and cleanly separates the "
            "top cluster of models."
        ),
        (
            f"Key visual references for this section are `{PLOT_REFERENCES['statistical']}`, "
            f"`{PLOT_REFERENCES['ml']}`, `{PLOT_REFERENCES['leaderboard']}`, "
            f"`{PLOT_REFERENCES['rnn_loss']}`, `{PLOT_REFERENCES['advanced']}`, and "
            f"`{PLOT_REFERENCES['ensemble']}`."
        ),
        "",
        "## 4. Results by Model Family",
        "",
        (
            f"The family summary below shows the approved 22-model study. Ensemble models are best on average "
            f"(mean RMSE={family_summary.loc[family_summary['Family'] == 'Ensemble', 'Mean RMSE'].iloc[0]:.4f}), "
            f"but the gap to ML is small. Statistical models remain highly competitive, while both deep-learning "
            "families trail by a wide margin."
        ),
        "",
        format_family_summary_table(family_summary),
        "",
        (
            f"Statistical baselines were unexpectedly strong. `{best_statistical['model_name']}` and its SARIMA "
            f"counterpart tied at RMSE={float(best_statistical['RMSE']):.4f}, only about {winner_vs_arima_pct:.2f}% "
            "worse than the overall winner. This indicates that the series retains substantial autocorrelation and "
            "can be forecast well with low-parameter temporal models."
        ),
        (
            f"Within ML, `{best_ml['model_name']}` and `LinearRegression` formed a near tie around RMSE=0.1569. "
            "The linear SVR and Lasso variants also stayed in the same cluster, while the nonlinear tree-based "
            "models deteriorated markedly: LightGBM finished at RMSE=0.2061, RandomForest at 0.2171, and XGBoost at 0.2207."
        ),
        (
            f"Advanced DL improved on the simple RNN step but still did not threaten the leading families. "
            f"`{best_advanced_dl['model_name']}` reached RMSE={float(best_advanced_dl['RMSE']):.4f} and "
            f"R^2={float(best_advanced_dl['R2']):.4f}, clearly better than the univariate RNNs but still well behind "
            "the linear, statistical, and ensemble leaders."
        ),
        (
            f"The RNN family was the weakest approved family by a large margin. `{best_rnn['model_name']}` was still "
            f"at RMSE={float(best_rnn['RMSE']):.4f}, and all three approved RNNs had negative R^2, which means they "
            "performed worse than a constant-mean baseline on the test set."
        ),
        "",
        "### Full Approved Leaderboard",
        "",
        format_official_leaderboard_table(leaderboard),
        "",
        "## 5. Key Findings & Ablation Insights",
        "",
        (
            f"1. Ensembles only barely improve the best base learner. The winning ensemble beats `{runner_up['model_name']}` "
            f"by {winner_vs_runner_up:.6f} RMSE, which is a relative margin of only "
            f"{winner_vs_runner_up / float(runner_up['RMSE']) * 100.0:.3f}%. The ensemble is best, but the family-level "
            "conclusion is more important than the exact first-place row."
        ),
        (
            "2. The most informative signals are recent persistence and short-horizon trend. A Ridge(alpha=1.0) refit "
            "on the saved standardized train split shows that `lag_1`, `rolling_mean_7`, `rolling_mean_30`, `lag_2`, "
            "and `lag_5` carry the largest coefficients. Calendar variables do not appear in the top features, which "
            "suggests that the study is mostly extracting autoregressive structure rather than seasonal calendar effects."
        ),
        "",
        format_feature_table(feature_importance),
        "",
        (
            "3. Tree-based nonlinear models underperform because they struggle with smooth extrapolation on a trending "
            "holdout window. Their forecasts appear competitive in-sample, but out-of-sample they flatten relative to "
            "the linear and statistical families."
        ),
        (
            "4. Multivariate advanced DL is better than univariate RNNs, which means the engineered feature set is "
            "useful even for neural models. However, the sample size is still too limited to justify the additional "
            "capacity relative to the strong low-parameter baselines."
        ),
        (
            "5. The broad picture is robust even when the wider archived sweep is considered: simpler families keep "
            "winning, while depth, attention, and stacking only help when tightly constrained."
        ),
        "",
        "## 6. Statistical Analysis",
        "",
        (
            "The top three approved models are all very close to one another, so the correct interpretation is "
            "practical parity rather than a dramatic win. The table below compares their RMSE values against the "
            "naive persistence baseline and the best statistical baseline."
        ),
        "",
        format_improvement_table(improvement_summary),
        "",
        (
            f"The winner improves RMSE over naive persistence by {winner_vs_naive_pct:.2f}% and over the best "
            f"statistical baseline by {winner_vs_arima_pct:.2f}%. These are real but modest gains. The difference "
            f"between first and second place is only {winner_vs_runner_up:.6f} RMSE, and the gap between second and "
            f"third is only {runner_up_vs_third:.6f}."
        ),
        (
            "Given that the study uses one chronological holdout instead of multiple rolling windows, and given that "
            "no Diebold-Mariano or bootstrap significance test was archived with the results, it would be overstated "
            "to claim that the winner is statistically superior to the runner-up in a formal inferential sense. What "
            "is clearly meaningful is the family-level pattern: the top cluster of ensemble, linear, and ARIMA-style "
            "models is materially better than naive persistence and far better than the approved RNN family."
        ),
        "",
        "## 7. Limitations & Future Work",
        "",
        "- The final ranking is based on a single fixed holdout from 2022-04-26 to 2024-12-31 rather than repeated walk-forward backtests.",
        "- No formal forecast-difference significance test was saved, so the top-3 ranking should be treated as practically close.",
        "- The approved deep-learning steps were intentionally compact. Broader tuning, longer training schedules, or richer exogenous inputs could change the DL ordering.",
        "- The target is forecast from lagged index behavior and calendar fields only; macro, sentiment, and order-book features were not included.",
        "- Feature importance was derived from a step-9 Ridge refit on saved training arrays because trained step-5 model objects were not persisted as artifacts.",
        "Future work should prioritize rolling-origin evaluation, ARIMAX or SARIMAX with exogenous regressors, regime-aware ensembles, probabilistic intervals, and stronger DL tuning only after the linear/statistical ceiling is genuinely challenged.",
        "",
        "## 8. Appendix",
        "",
        (
            "The appendix inventories the broader 42-model archived sweep retained in the repository at "
            "`results/MASTER_LEADERBOARD.csv`. This appendix is included for reproducibility because the workspace "
            "still contains those experimental artifacts. The official conclusions above do not rank those archived "
            "rows against the approved 22-row step-8 leaderboard."
        ),
        (
            "For the archived neural variants, the common training shorthand is: lookback=30, Adam(lr=0.001), "
            "MSE loss, batch_size=32, temporal validation split, and early stopping unless explicitly noted otherwise."
        ),
        "",
        format_appendix_tables(master_leaderboard),
        "",
    ]
    return "\n".join(report_lines).rstrip() + "\n"


def build_study_complete_section(
    leaderboard: pd.DataFrame,
    feature_importance: pd.DataFrame,
    improvement_summary: pd.DataFrame,
) -> str:
    winner = leaderboard.iloc[0]
    runner_up = leaderboard.iloc[1]
    third_place = leaderboard.iloc[2]
    best_statistical = leaderboard.loc[leaderboard["model_family"] == "statistical"].iloc[0]
    best_feature = feature_importance.iloc[0]
    top3_mean = improvement_summary.loc[
        improvement_summary["Model"] == "Top-3 mean RMSE", "RMSE"
    ].iloc[0]

    lines = [
        LOG_SECTION_HEADER,
        "",
        "### Completion Summary",
        "- Final report written to `FINAL_REPORT.md`.",
        (
            f"- Winner: `{winner['model_name']}` ({winner['family_label']}) with RMSE={float(winner['RMSE']):.4f}, "
            f"MAE={float(winner['MAE']):.4f}, and R^2={float(winner['R2']):.4f}."
        ),
        (
            f"- Runner-up: `{runner_up['model_name']}` with RMSE={float(runner_up['RMSE']):.4f}; "
            f"third place `{third_place['model_name']}` followed at RMSE={float(third_place['RMSE']):.4f}."
        ),
        (
            f"- Best statistical baseline: `{best_statistical['model_name']}` at RMSE={float(best_statistical['RMSE']):.4f}, "
            "showing that the overall winner only edges the best low-parameter temporal model."
        ),
        (
            f"- Key ablation takeaway: the top-3 mean RMSE is {float(top3_mean):.6f}, so the exact winner is less important "
            "than the stable result that ensemble, linear, and ARIMA-style models dominate the leaderboard."
        ),
        (
            f"- Feature takeaway: `{best_feature['feature']}` had the largest standardized Ridge coefficient "
            f"({float(best_feature['abs_coef']):.4f}), reinforcing that short-lag persistence is the main predictive signal."
        ),
        "- Deep learning improved when multivariate engineered features were used, but both approved DL families remained behind the best statistical and ML models.",
    ]
    return "\n".join(lines)


def upsert_log_section(section_text: str) -> None:
    current_text = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    pattern = rf"{re.escape(LOG_SECTION_HEADER)}.*?(?=\n## |\Z)"

    if re.search(pattern, current_text, flags=re.DOTALL):
        updated_text = re.sub(
            pattern,
            section_text.rstrip(),
            current_text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        separator = "\n\n" if current_text.strip() else ""
        updated_text = f"{current_text.rstrip()}{separator}{section_text.rstrip()}\n"

    RESEARCH_LOG_PATH.write_text(updated_text.rstrip() + "\n", encoding="utf-8")


def run_final_report() -> tuple[Path, int]:
    split_info = load_split_info()
    official_leaderboard = load_official_leaderboard()
    archived_master = load_archived_master_leaderboard()
    family_summary = build_family_summary(official_leaderboard)
    feature_importance = compute_ridge_feature_importance()
    improvement_summary = build_improvement_summary(official_leaderboard)

    report_text = build_final_report(
        split_info=split_info,
        leaderboard=official_leaderboard,
        family_summary=family_summary,
        feature_importance=feature_importance,
        improvement_summary=improvement_summary,
        master_leaderboard=archived_master,
    )
    FINAL_REPORT_PATH.write_text(report_text, encoding="utf-8")

    study_complete_section = build_study_complete_section(
        leaderboard=official_leaderboard,
        feature_importance=feature_importance,
        improvement_summary=improvement_summary,
    )
    upsert_log_section(study_complete_section)

    word_count = count_words(report_text)
    return FINAL_REPORT_PATH, word_count


def main() -> None:
    output_path, word_count = run_final_report()
    print(f"Wrote final report to {output_path} ({word_count} words).")
    print(f"Updated research log at {RESEARCH_LOG_PATH}.")


if __name__ == "__main__":
    main()

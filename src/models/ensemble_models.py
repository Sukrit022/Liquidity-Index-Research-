from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics_tracker import REGISTRY_PATH, compute_metrics, load_registry, log_result


RESULTS_ROOT = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions" / "ensemble"
LEADERBOARD_CSV_PATH = RESULTS_ROOT / "leaderboard.csv"
LEADERBOARD_MD_PATH = RESULTS_ROOT / "LEADERBOARD.md"
PLOT_PATH = PROJECT_ROOT / "plots" / "leaderboard_comparison.png"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

LOG_SECTION_HEADER = "## Ensemble Models"
STACKING_WARMUP = 30
SORT_COLUMNS = ["RMSE", "MAE", "MAPE", "model_family", "model_name"]
FAMILY_ORDER = ["statistical", "ml", "dl", "dl_advanced", "ensemble"]
FAMILY_LABELS = {
    "statistical": "Statistical",
    "ml": "ML",
    "dl": "Deep Learning RNN",
    "dl_advanced": "Advanced DL",
    "ensemble": "Ensemble",
}
FAMILY_COLORS = {
    "statistical": "#4E79A7",
    "ml": "#F28E2B",
    "dl": "#59A14F",
    "dl_advanced": "#E15759",
    "ensemble": "#B07AA1",
}

SIMPLE_AVERAGE_NAME = "Simple Average Ensemble (Top-3 ML)"
WEIGHTED_NAME = "Weighted Ensemble (Top-5 Overall)"
STACKING_NAME = "Stacking Ensemble (Linear Meta-Learner)"


@dataclass(frozen=True)
class SelectedModel:
    model_family: str
    model_name: str
    predictions_path: Path
    rmse: float
    mae: float
    mape: float
    slug: str


@dataclass(frozen=True)
class EnsembleResult:
    model_name: str
    slug: str
    metrics: dict[str, float]
    predictions_path: Path


def ensure_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)


def sort_registry(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        SORT_COLUMNS,
        ascending=[True] * len(SORT_COLUMNS),
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def prune_existing_ensemble_rows() -> None:
    if not REGISTRY_PATH.exists():
        return

    registry = load_registry()
    filtered = registry.loc[registry["model_family"] != "ensemble"].copy()
    filtered.to_csv(REGISTRY_PATH, index=False)


def load_base_registry() -> pd.DataFrame:
    registry = load_registry().copy()
    registry = registry.loc[registry["model_family"] != "ensemble"].copy()
    registry = registry.dropna(subset=["predictions_path"])
    registry = sort_registry(registry)

    if registry.empty:
        raise FileNotFoundError(
            "No base-model entries were found in results/metrics_registry.csv."
        )

    return registry


def to_selected_models(frame: pd.DataFrame) -> list[SelectedModel]:
    models: list[SelectedModel] = []
    for row in frame.itertuples(index=False):
        predictions_path = PROJECT_ROOT / str(row.predictions_path)
        slug = predictions_path.stem.removesuffix("_predictions")
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file referenced by registry: {predictions_path}")

        models.append(
            SelectedModel(
                model_family=str(row.model_family),
                model_name=str(row.model_name),
                predictions_path=predictions_path,
                rmse=float(row.RMSE),
                mae=float(row.MAE),
                mape=float(row.MAPE),
                slug=slug,
            )
        )
    return models


def select_top_models(
    registry: pd.DataFrame,
    count: int,
    family: str | None = None,
) -> list[SelectedModel]:
    selected = registry.copy()
    if family is not None:
        selected = selected.loc[selected["model_family"] == family].copy()

    selected = sort_registry(selected).head(count)
    if len(selected) < count:
        scope = f"family '{family}'" if family is not None else "all model families"
        raise ValueError(f"Expected at least {count} models in {scope}, found {len(selected)}.")

    return to_selected_models(selected)


def load_prediction_frame(predictions_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    required_columns = {"date", "actual", "prediction"}
    if not required_columns.issubset(frame.columns):
        missing = ", ".join(sorted(required_columns - set(frame.columns)))
        raise ValueError(f"{predictions_path} is missing required columns: {missing}")

    loaded = frame.loc[:, ["date", "actual", "prediction"]].copy()
    loaded["date"] = pd.to_datetime(loaded["date"], errors="raise")
    loaded["actual"] = pd.to_numeric(loaded["actual"], errors="raise").astype(np.float64)
    loaded["prediction"] = pd.to_numeric(loaded["prediction"], errors="raise").astype(np.float64)
    return loaded.sort_values("date").reset_index(drop=True)


def load_aligned_predictions(models: list[SelectedModel]) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None

    for model in models:
        model_frame = load_prediction_frame(model.predictions_path).rename(
            columns={"prediction": model.slug}
        )

        if aligned is None:
            aligned = model_frame.rename(columns={"actual": "actual"})
            continue

        if len(model_frame) != len(aligned):
            raise ValueError(
                "Prediction horizon mismatch: "
                f"{model.predictions_path} has {len(model_frame)} rows, expected {len(aligned)}."
            )

        if not model_frame["date"].equals(aligned["date"]):
            raise ValueError(f"Date alignment mismatch for {model.predictions_path}.")

        if not np.allclose(
            model_frame["actual"].to_numpy(dtype=np.float64),
            aligned["actual"].to_numpy(dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(f"Actual-value mismatch for {model.predictions_path}.")

        aligned[model.slug] = model_frame[model.slug].to_numpy(dtype=np.float64)

    if aligned is None:
        raise ValueError("No prediction frames were loaded.")

    return aligned


def build_simple_average_predictions(frame: pd.DataFrame, models: list[SelectedModel]) -> np.ndarray:
    return frame[[model.slug for model in models]].mean(axis=1).to_numpy(dtype=np.float64)


def build_inverse_rmse_weights(models: list[SelectedModel]) -> dict[str, float]:
    inverse_rmse = np.asarray(
        [1.0 / max(model.rmse, 1e-12) for model in models],
        dtype=np.float64,
    )
    normalized = inverse_rmse / inverse_rmse.sum()
    return {
        model.slug: float(weight)
        for model, weight in zip(models, normalized, strict=True)
    }


def build_weighted_predictions(
    frame: pd.DataFrame,
    models: list[SelectedModel],
) -> tuple[np.ndarray, dict[str, float]]:
    weights = build_inverse_rmse_weights(models)
    weighted = np.zeros(len(frame), dtype=np.float64)
    for model in models:
        weighted += frame[model.slug].to_numpy(dtype=np.float64) * weights[model.slug]
    return weighted, weights


def build_stacking_predictions(
    frame: pd.DataFrame,
    models: list[SelectedModel],
    warmup: int = STACKING_WARMUP,
) -> tuple[np.ndarray, int]:
    feature_matrix = frame[[model.slug for model in models]].to_numpy(dtype=np.float64)
    target = frame["actual"].to_numpy(dtype=np.float64)
    if len(feature_matrix) < 2:
        raise ValueError("Stacking requires at least two aligned observations.")

    effective_warmup = min(max(warmup, feature_matrix.shape[1] + 1), len(feature_matrix) - 1)
    predictions = np.full(len(feature_matrix), np.nan, dtype=np.float64)

    for index in range(effective_warmup, len(feature_matrix)):
        meta_model = LinearRegression()
        meta_model.fit(feature_matrix[:index], target[:index])
        predictions[index] = float(meta_model.predict(feature_matrix[index : index + 1])[0])

    predictions[:effective_warmup] = feature_matrix[:effective_warmup].mean(axis=1)
    return predictions, effective_warmup


def save_predictions(
    slug: str,
    dates: pd.Series,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> Path:
    output_path = PREDICTIONS_DIR / f"{slug}_predictions.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
            "actual": actual,
            "prediction": prediction,
            "residual": actual - prediction,
            "abs_error": np.abs(actual - prediction),
        }
    )
    frame.to_csv(output_path, index=False)
    return output_path


def persist_ensemble_result(
    model_name: str,
    slug: str,
    dates: pd.Series,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> EnsembleResult:
    metrics = compute_metrics(actual, prediction)
    predictions_path = save_predictions(
        slug=slug,
        dates=dates,
        actual=actual,
        prediction=prediction,
    )
    log_result("ensemble", model_name, metrics, predictions_path)
    return EnsembleResult(
        model_name=model_name,
        slug=slug,
        metrics=metrics,
        predictions_path=predictions_path,
    )


def build_leaderboard_frame() -> pd.DataFrame:
    leaderboard = sort_registry(load_registry().copy())
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1, dtype=np.int64))
    leaderboard["family_label"] = leaderboard["model_family"].map(FAMILY_LABELS).fillna(
        leaderboard["model_family"]
    )
    return leaderboard


def format_markdown_table(leaderboard: pd.DataFrame) -> str:
    lines = [
        "| Rank | Family | Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R^2 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.family_label} | {row.model_name} | "
            f"{row.RMSE:.4f} | {row.MAE:.4f} | {row.MAPE:.4f} | {row.SMAPE:.4f} | {row.R2:.4f} |"
        )
    return "\n".join(lines)


def truncate_label(label: str, max_length: int = 54) -> str:
    if len(label) <= max_length:
        return label
    return label[: max_length - 3].rstrip() + "..."


def render_comparison_plot(leaderboard: pd.DataFrame) -> None:
    families_present = [
        family for family in FAMILY_ORDER if family in set(leaderboard["model_family"])
    ]
    if not families_present:
        raise ValueError("No families were available for the leaderboard plot.")

    figure_height = max(10.0, 2.4 * len(families_present) + 0.45 * len(leaderboard))
    figure, axes = plt.subplots(
        nrows=len(families_present),
        ncols=1,
        figsize=(16, figure_height),
        sharex=True,
    )

    if len(families_present) == 1:
        axes = [axes]

    best_rmse = float(leaderboard["RMSE"].min())

    for axis, family in zip(axes, families_present, strict=True):
        family_frame = leaderboard.loc[leaderboard["model_family"] == family].copy()
        family_frame = family_frame.sort_values(["RMSE", "MAE", "MAPE"], ascending=True)
        labels = [truncate_label(name) for name in family_frame["model_name"]]
        bars = axis.barh(
            labels,
            family_frame["RMSE"],
            color=FAMILY_COLORS.get(family, "#7F7F7F"),
            alpha=0.9,
        )

        axis.invert_yaxis()
        axis.set_title(f"{FAMILY_LABELS.get(family, family)} ({len(family_frame)} models)")
        axis.grid(axis="x", alpha=0.25)
        axis.axvline(best_rmse, color="#333333", linestyle="--", linewidth=1.0)

        for bar, row in zip(bars, family_frame.itertuples(index=False), strict=True):
            axis.text(
                float(bar.get_width()) + 0.002,
                float(bar.get_y()) + float(bar.get_height()) / 2.0,
                f"#{int(row.rank)}  {float(row.RMSE):.4f}",
                va="center",
                fontsize=8,
            )

    axes[-1].set_xlabel("RMSE (lower is better)")
    figure.suptitle("Liquidity Index Ablation Leaderboard by Model Family", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    figure.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_leaderboard_markdown(
    leaderboard: pd.DataFrame,
    best_overall: pd.Series,
    best_ensemble: pd.Series,
) -> None:
    content = "\n".join(
        [
            "# Full Ablation Leaderboard",
            "",
            "Generated from `results/metrics_registry.csv` after step-8 ensemble evaluation.",
            "",
            (
                f"Best overall model: `{best_overall['model_name']}` "
                f"({FAMILY_LABELS.get(str(best_overall['model_family']), str(best_overall['model_family']))}) "
                f"with RMSE={float(best_overall['RMSE']):.4f}."
            ),
            (
                f"Best ensemble: `{best_ensemble['model_name']}` "
                f"with RMSE={float(best_ensemble['RMSE']):.4f}."
            ),
            "",
            format_markdown_table(leaderboard),
            "",
            f"Comparison plot: `plots/{PLOT_PATH.name}`",
            "",
        ]
    )
    LEADERBOARD_MD_PATH.write_text(content, encoding="utf-8")


def format_model_list(models: list[SelectedModel]) -> str:
    return ", ".join(f"`{model.model_name}`" for model in models)


def format_weight_summary(
    models: list[SelectedModel],
    weights: dict[str, float],
) -> str:
    parts = []
    for model in models:
        parts.append(f"`{model.model_name}`={weights[model.slug]:.3f}")
    return ", ".join(parts)


def build_research_log_section(
    leaderboard: pd.DataFrame,
    simple_models: list[SelectedModel],
    weighted_models: list[SelectedModel],
    weighted_weights: dict[str, float],
    stacking_models: list[SelectedModel],
    stacking_warmup: int,
) -> str:
    best_overall = leaderboard.iloc[0]
    base_leaderboard = leaderboard.loc[leaderboard["model_family"] != "ensemble"].copy()
    best_base = base_leaderboard.iloc[0]
    ensemble_leaderboard = leaderboard.loc[leaderboard["model_family"] == "ensemble"].copy()
    best_ensemble = ensemble_leaderboard.iloc[0]
    rmse_delta = float(best_base["RMSE"]) - float(best_ensemble["RMSE"])

    if rmse_delta > 0:
        ensemble_takeaway = (
            f"- Best ensemble vs best base model: ensemble improved RMSE by {rmse_delta:.4f}."
        )
    elif rmse_delta < 0:
        ensemble_takeaway = (
            f"- Best ensemble vs best base model: ensemble trailed by {-rmse_delta:.4f} RMSE."
        )
    else:
        ensemble_takeaway = "- Best ensemble matched the best base model on RMSE."

    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Ensemble Setup",
        f"- Source of truth: `{REGISTRY_PATH.relative_to(PROJECT_ROOT).as_posix()}` and `results/predictions/**`.",
        f"- Simple Average Ensemble used the top-3 ML models by RMSE: {format_model_list(simple_models)}.",
        (
            "- Weighted Ensemble used the top-5 non-ensemble models across all families "
            f"with inverse-RMSE weights: {format_weight_summary(weighted_models, weighted_weights)}."
        ),
        (
            "- Stacking Ensemble used a LinearRegression meta-learner on the top-3 "
            "non-ensemble base models with walk-forward out-of-fold predictions "
            f"after a {stacking_warmup}-observation warm-up window."
        ),
        "",
        "### Ensemble Results",
        format_markdown_table(ensemble_leaderboard),
        "",
        "### Full Leaderboard",
        format_markdown_table(leaderboard),
        "",
        "### Findings",
        (
            f"- Best overall model after ensembles: `{best_overall['model_name']}` "
            f"({best_overall['family_label']}) with RMSE={float(best_overall['RMSE']):.4f}, "
            f"MAE={float(best_overall['MAE']):.4f}, and R^2={float(best_overall['R2']):.4f}."
        ),
        (
            f"- Best ensemble: `{best_ensemble['model_name']}` with RMSE={float(best_ensemble['RMSE']):.4f}, "
            f"MAE={float(best_ensemble['MAE']):.4f}, and R^2={float(best_ensemble['R2']):.4f}."
        ),
        ensemble_takeaway,
        (
            f"- Weighted ensemble constituents: {format_model_list(weighted_models)}."
        ),
        (
            f"- Stacking base models: {format_model_list(stacking_models)}."
        ),
        "",
        "### Saved Artifacts",
        "- `src/models/ensemble_models.py`",
        "- `results/predictions/ensemble/*.csv`",
        "- `results/leaderboard.csv`",
        "- `results/LEADERBOARD.md`",
        "- `plots/leaderboard_comparison.png`",
        "- `results/metrics_registry.csv`",
    ]
    return "\n".join(section_lines)


def upsert_research_log(section_text: str) -> None:
    current_text = (
        RESEARCH_LOG_PATH.read_text(encoding="utf-8")
        if RESEARCH_LOG_PATH.exists()
        else ""
    )
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


def run_all_ensemble_models() -> pd.DataFrame:
    ensure_dirs()
    prune_existing_ensemble_rows()

    base_registry = load_base_registry()
    simple_models = select_top_models(base_registry, count=3, family="ml")
    weighted_models = select_top_models(base_registry, count=5)
    stacking_models = select_top_models(base_registry, count=3)

    simple_frame = load_aligned_predictions(simple_models)
    weighted_frame = load_aligned_predictions(weighted_models)
    stacking_frame = load_aligned_predictions(stacking_models)

    actual = simple_frame["actual"].to_numpy(dtype=np.float64)
    dates = simple_frame["date"]

    if not dates.equals(weighted_frame["date"]) or not dates.equals(stacking_frame["date"]):
        raise ValueError("Ensemble component predictions are not aligned on identical dates.")

    if not np.allclose(
        actual,
        weighted_frame["actual"].to_numpy(dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    ) or not np.allclose(
        actual,
        stacking_frame["actual"].to_numpy(dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Ensemble component predictions do not share the same actual series.")

    simple_predictions = build_simple_average_predictions(simple_frame, simple_models)
    weighted_predictions, weighted_weights = build_weighted_predictions(
        weighted_frame,
        weighted_models,
    )
    stacking_predictions, stacking_warmup = build_stacking_predictions(
        stacking_frame,
        stacking_models,
    )

    ensemble_results = [
        persist_ensemble_result(
            model_name=SIMPLE_AVERAGE_NAME,
            slug="simple_average_top3_ml",
            dates=dates,
            actual=actual,
            prediction=simple_predictions,
        ),
        persist_ensemble_result(
            model_name=WEIGHTED_NAME,
            slug="weighted_top5_overall",
            dates=dates,
            actual=actual,
            prediction=weighted_predictions,
        ),
        persist_ensemble_result(
            model_name=STACKING_NAME,
            slug="stacking_linear_top3_base",
            dates=dates,
            actual=actual,
            prediction=stacking_predictions,
        ),
    ]

    leaderboard = build_leaderboard_frame()
    leaderboard.to_csv(LEADERBOARD_CSV_PATH, index=False)
    render_comparison_plot(leaderboard)

    best_overall = leaderboard.iloc[0]
    best_ensemble = leaderboard.loc[leaderboard["model_family"] == "ensemble"].iloc[0]
    write_leaderboard_markdown(leaderboard, best_overall, best_ensemble)
    upsert_research_log(
        build_research_log_section(
            leaderboard=leaderboard,
            simple_models=simple_models,
            weighted_models=weighted_models,
            weighted_weights=weighted_weights,
            stacking_models=stacking_models,
            stacking_warmup=stacking_warmup,
        )
    )

    results_frame = pd.DataFrame(
        [
            {
                "model_name": result.model_name,
                "slug": result.slug,
                "RMSE": result.metrics["RMSE"],
                "MAE": result.metrics["MAE"],
                "R2": result.metrics["R2"],
                "predictions_path": result.predictions_path.relative_to(PROJECT_ROOT).as_posix(),
            }
            for result in ensemble_results
        ]
    ).sort_values(["RMSE", "MAE"], ascending=True)

    return results_frame.reset_index(drop=True)


def main() -> None:
    summary = run_all_ensemble_models()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

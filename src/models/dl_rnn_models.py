from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_CANDIDATES = (
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "artifacts",
)
RESULTS_ROOT = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions" / "dl"
PLOTS_DIR = PROJECT_ROOT / "plots" / "dl"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

LOOKBACK_WINDOW = 30
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
VALIDATION_FRACTION = 0.1
LEARNING_RATE = 1e-3
SEED = 42
LOG_SECTION_HEADER = "## Deep Learning RNN Models"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    slug: str
    factory: Callable[[tuple[int, int]], keras.Model]


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    slug: str
    predictions: np.ndarray
    metrics: dict[str, float]
    predictions_path: Path
    history: dict[str, list[float]]


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


def load_inputs() -> dict[str, Any]:
    arrays_path = resolve_artifact_path("preprocessed_arrays.joblib")
    scalers_path = resolve_artifact_path("minmax_scalers.joblib")

    arrays = joblib.load(arrays_path)
    scalers = joblib.load(scalers_path)
    if not isinstance(scalers, dict) or "target" not in scalers:
        raise ValueError("minmax_scalers.joblib must contain a 'target' MinMaxScaler.")

    return {
        "arrays_path": arrays_path,
        "scalers_path": scalers_path,
        "train_scaled": np.asarray(arrays["y_train_dl_scaled"], dtype=np.float32).reshape(-1),
        "test_scaled": np.asarray(arrays["y_test_dl_scaled"], dtype=np.float32).reshape(-1),
        "train_actual": np.asarray(arrays["y_train"], dtype=np.float64).reshape(-1),
        "test_actual": np.asarray(arrays["y_test"], dtype=np.float64).reshape(-1),
        "train_index": pd.to_datetime(arrays["train_index"]),
        "test_index": pd.to_datetime(arrays["test_index"]),
        "target_scaler": scalers["target"],
        "target_name": str(arrays.get("target_name", "liquidity_index")),
    }


def build_univariate_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    if values.size <= lookback:
        raise ValueError(
            f"Need more than {lookback} observations to build sequences; received {values.size}."
        )

    X_values = []
    y_values = []
    for idx in range(lookback, values.size):
        X_values.append(values[idx - lookback:idx])
        y_values.append(values[idx])

    X_array = np.asarray(X_values, dtype=np.float32).reshape(-1, lookback, 1)
    y_array = np.asarray(y_values, dtype=np.float32).reshape(-1)
    return X_array, y_array


def build_test_sequences(
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([train_scaled, test_scaled]).astype(np.float32)
    test_start = len(train_scaled)

    X_values = []
    y_values = []
    for idx in range(test_start, combined.size):
        X_values.append(combined[idx - lookback:idx])
        y_values.append(combined[idx])

    X_array = np.asarray(X_values, dtype=np.float32).reshape(-1, lookback, 1)
    y_array = np.asarray(y_values, dtype=np.float32).reshape(-1)
    return X_array, y_array


def temporal_train_validation_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_size = max(1, int(len(X_train) * validation_fraction))
    if validation_size >= len(X_train):
        raise ValueError("validation split leaves no training samples.")

    split_index = len(X_train) - validation_size
    return (
        X_train[:split_index],
        X_train[split_index:],
        y_train[:split_index],
        y_train[split_index:],
    )


def build_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, dropout=0.2),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def build_bidirectional_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(64, dropout=0.2)),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def build_gru_model(input_shape: tuple[int, int]) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GRU(64, dropout=0.2),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_name="LSTM (64 units, dropout=0.2)",
            slug="lstm",
            factory=build_lstm_model,
        ),
        ModelSpec(
            model_name="Bidirectional LSTM (64 units, dropout=0.2)",
            slug="bidirectional_lstm",
            factory=build_bidirectional_lstm_model,
        ),
        ModelSpec(
            model_name="GRU (64 units, dropout=0.2)",
            slug="gru",
            factory=build_gru_model,
        ),
    ]


def inverse_transform_target(values: np.ndarray, target_scaler: Any) -> np.ndarray:
    return target_scaler.inverse_transform(
        np.asarray(values, dtype=np.float64).reshape(-1, 1)
    ).reshape(-1)


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


def plot_training_loss(result: ModelResult) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(result.history.get("loss", []), label="train_loss", linewidth=2.0)
    axis.plot(result.history.get("val_loss", []), label="val_loss", linewidth=2.0)
    axis.set_title(f"{result.model_name} Loss Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()

    output_path = PLOTS_DIR / f"{result.slug}_loss_curve.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
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
    train_sequence_count = len(payload["X_train_full"])
    validation_size = len(payload["X_val"])

    lines = [
        LOG_SECTION_HEADER,
        "",
        "### Evaluation Setup",
        f"- Source arrays: `{payload['arrays_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Source scaler bundle: `{payload['scalers_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Train window: {train_index.min().date()} to {train_index.max().date()} ({len(train_index)} rows)",
        f"- Test window: {test_index.min().date()} to {test_index.max().date()} ({len(test_index)} rows)",
        f"- Target series: `{payload['target_name']}` scaled with `MinMaxScaler`; input sequences are univariate lookback windows of length {LOOKBACK_WINDOW}.",
        f"- Training sequences: {train_sequence_count} total, with the last {validation_size} sequences reserved as a temporal validation split.",
        f"- Training config: Adam(lr={LEARNING_RATE}), MSE loss, epochs={EPOCHS}, batch_size={BATCH_SIZE}, EarlyStopping(patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True).",
        "- Models evaluated: Vanilla LSTM, Bidirectional LSTM, and GRU with 64 recurrent units and dropout=0.2.",
        "",
        "### Test Metrics",
        format_metrics_table(results_frame),
        "",
        "### Findings",
        f"- Best DL model by RMSE: `{best_row.model_name}` with RMSE={best_row.RMSE:.4f}, MAE={best_row.MAE:.4f}, and R^2={best_row.R2:.4f}.",
        f"- Weakest DL model by RMSE: `{worst_row.model_name}` with RMSE={worst_row.RMSE:.4f}.",
        "- Using only the scaled target history isolates sequence-model behavior from the engineered lag and rolling features used by the classical ML step.",
        "- Prediction CSVs were written to `results/predictions/dl/`, loss curves were written to `plots/dl/`, and all three models were registered through `utils/metrics_tracker.py`.",
        "",
        "### Saved Artifacts",
        "- `src/models/dl_rnn_models.py`",
        "- `results/predictions/dl/*.csv`",
        "- `plots/dl/*_loss_curve.png`",
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


def train_single_model(
    spec: ModelSpec,
    payload: dict[str, Any],
) -> ModelResult:
    from utils.metrics_tracker import compute_metrics, log_result

    tf.keras.backend.clear_session()
    model = spec.factory((LOOKBACK_WINDOW, 1))
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        payload["X_train"],
        payload["y_train"],
        validation_data=(payload["X_val"], payload["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False,
    )

    predicted_scaled = model.predict(payload["X_test"], verbose=0).reshape(-1)
    predicted_actual = inverse_transform_target(predicted_scaled, payload["target_scaler"])
    predictions_path = save_predictions(
        slug=spec.slug,
        dates=payload["test_index"],
        actual=payload["test_actual"],
        predicted=predicted_actual,
    )
    metrics = compute_metrics(payload["test_actual"], predicted_actual)
    log_result("dl", spec.model_name, metrics, predictions_path)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    return ModelResult(
        model_name=spec.model_name,
        slug=spec.slug,
        predictions=predicted_actual,
        metrics=metrics,
        predictions_path=predictions_path,
        history=history_dict,
    )


def prepare_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    X_train_full, y_train_full = build_univariate_sequences(raw_payload["train_scaled"], LOOKBACK_WINDOW)
    X_test, y_test_scaled = build_test_sequences(
        raw_payload["train_scaled"],
        raw_payload["test_scaled"],
        LOOKBACK_WINDOW,
    )
    X_train, X_val, y_train, y_val = temporal_train_validation_split(X_train_full, y_train_full)

    if len(X_test) != len(raw_payload["test_actual"]):
        raise ValueError(
            "Test sequence count does not align with test targets: "
            f"{len(X_test)} sequences vs {len(raw_payload['test_actual'])} targets."
        )

    if len(y_test_scaled) != len(raw_payload["test_actual"]):
        raise ValueError(
            "Scaled test target count does not align with raw test targets: "
            f"{len(y_test_scaled)} vs {len(raw_payload['test_actual'])}."
        )

    return {
        **raw_payload,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": X_test,
        "y_test_scaled": y_test_scaled,
    }


def run_all_rnn_models() -> pd.DataFrame:
    ensure_dirs()
    payload = prepare_payload(load_inputs())

    results: list[ModelResult] = []
    for spec in get_model_specs():
        result = train_single_model(spec, payload)
        plot_training_loss(result)
        results.append(result)

    results_frame = build_results_frame(results)
    upsert_research_log(build_research_log_section(results_frame, payload))
    return results_frame


def main() -> None:
    summary = run_all_rnn_models()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

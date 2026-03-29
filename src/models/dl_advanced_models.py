from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
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
PREDICTIONS_DIR = RESULTS_ROOT / "predictions" / "dl_advanced"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

LOOKBACK_WINDOW = 30
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
VALIDATION_FRACTION = 0.1
LEARNING_RATE = 1e-3
SEED = 42
LOG_SECTION_HEADER = "## Advanced DL Models"

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
    metrics: dict[str, float]
    predictions_path: Path
    architecture_path: Path
    history: dict[str, list[float]]


class DotProductAttention(layers.Layer):
    """Self-attention over sequence states using scaled dot-product weights."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        feature_dim = tf.cast(tf.shape(inputs)[-1], tf.float32)
        scores = tf.matmul(inputs, inputs, transpose_b=True)
        scaled_scores = scores / tf.math.sqrt(tf.maximum(feature_dim, 1.0))
        weights = tf.nn.softmax(scaled_scores, axis=-1)
        return tf.matmul(weights, inputs)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


def resolve_artifact_path(filename: str) -> Path:
    for base_dir in ARTIFACT_CANDIDATES:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path / filename) for path in ARTIFACT_CANDIDATES)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked}")


def ensure_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


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
        "train_feature_scaled": np.asarray(arrays["X_train_dl_scaled"], dtype=np.float32),
        "test_feature_scaled": np.asarray(arrays["X_test_dl_scaled"], dtype=np.float32),
        "train_target_scaled": np.asarray(arrays["y_train_dl_scaled"], dtype=np.float32).reshape(-1),
        "test_target_scaled": np.asarray(arrays["y_test_dl_scaled"], dtype=np.float32).reshape(-1),
        "train_actual": np.asarray(arrays["y_train"], dtype=np.float64).reshape(-1),
        "test_actual": np.asarray(arrays["y_test"], dtype=np.float64).reshape(-1),
        "train_index": pd.to_datetime(arrays["train_index"]),
        "test_index": pd.to_datetime(arrays["test_index"]),
        "target_scaler": scalers["target"],
        "target_name": str(arrays.get("target_name", "liquidity_index")),
        "feature_names": [str(name) for name in arrays.get("feature_names", [])],
    }


def build_feature_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_array = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
    if len(feature_array) <= lookback:
        raise ValueError(
            f"Need more than {lookback} observations to build sequences; received {len(feature_array)}."
        )
    if len(feature_array) != len(target_array):
        raise ValueError(
            "Feature and target arrays must have the same length; got "
            f"{len(feature_array)} and {len(target_array)}."
        )

    X_values = []
    y_values = []
    for idx in range(lookback, len(feature_array)):
        X_values.append(feature_array[idx - lookback : idx])
        y_values.append(target_array[idx])

    X_array = np.asarray(X_values, dtype=np.float32)
    y_array = np.asarray(y_values, dtype=np.float32).reshape(-1)
    return X_array, y_array


def build_test_sequences(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    combined_features = np.concatenate([train_features, test_features], axis=0).astype(
        np.float32
    )
    combined_targets = np.concatenate([train_targets, test_targets], axis=0).astype(
        np.float32
    )
    test_start = len(train_features)

    X_values = []
    y_values = []
    for idx in range(test_start, len(combined_features)):
        X_values.append(combined_features[idx - lookback : idx])
        y_values.append(combined_targets[idx])

    X_array = np.asarray(X_values, dtype=np.float32)
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


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm")
    return compile_model(model)


def build_attention_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = DotProductAttention(name="dot_product_attention")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="attention_lstm")
    return compile_model(model)


def build_temporal_transformer_model(input_shape: tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    attention_output = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        name="multi_head_attention",
    )(inputs, inputs)
    x = layers.Add()([inputs, attention_output])
    x = layers.LayerNormalization(name="attention_layer_norm")(x)
    x = layers.Dense(64, activation="relu", name="feed_forward")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="temporal_transformer",
    )
    return compile_model(model)


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_name="CNN-LSTM",
            slug="cnn_lstm",
            factory=build_cnn_lstm_model,
        ),
        ModelSpec(
            model_name="Attention LSTM",
            slug="attention_lstm",
            factory=build_attention_lstm_model,
        ),
        ModelSpec(
            model_name="Temporal Transformer",
            slug="temporal_transformer",
            factory=build_temporal_transformer_model,
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


def save_architecture_summary(
    model: keras.Model,
    spec: ModelSpec,
) -> Path:
    input_shape = tuple(int(dim) if dim is not None else -1 for dim in model.input_shape[1:])
    lines = [
        f"Model: {spec.model_name}",
        f"Slug: {spec.slug}",
        f"Input shape: {input_shape}",
        (
            "Training config: "
            f"Adam(lr={LEARNING_RATE}), loss=mse, epochs={EPOCHS}, "
            f"batch_size={BATCH_SIZE}, early_stopping_patience={EARLY_STOPPING_PATIENCE}"
        ),
        "",
        "Keras summary:",
    ]
    model.summary(print_fn=lines.append)
    output_path = RESULTS_ROOT / f"{spec.slug}_architecture.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
                "architecture_path": result.architecture_path.relative_to(PROJECT_ROOT).as_posix(),
                "epochs_trained": len(result.history.get("loss", [])),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["RMSE", "MAE", "MAPE"], ascending=True)
        .reset_index(drop=True)
    )


def format_metrics_table(results_frame: pd.DataFrame) -> str:
    lines = [
        "| # | Model | MAE | RMSE | MAPE (%) | SMAPE (%) | R^2 | Epochs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(results_frame.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.model_name} | {row.MAE:.4f} | {row.RMSE:.4f} | "
            f"{row.MAPE:.4f} | {row.SMAPE:.4f} | {row.R2:.4f} | {row.epochs_trained} |"
        )
    return "\n".join(lines)


def build_research_log_section(
    results_frame: pd.DataFrame,
    payload: dict[str, Any],
) -> str:
    best_row = results_frame.iloc[0]
    worst_row = results_frame.iloc[-1]
    train_index = payload["train_index"]
    test_index = payload["test_index"]
    train_sequence_count = len(payload["X_train_full"])
    validation_size = len(payload["X_val"])
    feature_count = payload["X_train"].shape[-1]

    lines = [
        LOG_SECTION_HEADER,
        "",
        "### Evaluation Setup",
        f"- Source arrays: `{payload['arrays_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Source scaler bundle: `{payload['scalers_path'].relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Train window: {train_index.min().date()} to {train_index.max().date()} ({len(train_index)} rows)",
        f"- Test window: {test_index.min().date()} to {test_index.max().date()} ({len(test_index)} rows)",
        (
            f"- Inputs: {feature_count} MinMax-scaled DL features from `artifacts/preprocessed_arrays.joblib`, "
            f"arranged as lookback-{LOOKBACK_WINDOW} sequences. Targets remain the scaled "
            f"`{payload['target_name']}` series for inverse-transformed evaluation."
        ),
        (
            f"- Training sequences: {train_sequence_count} total, with the last {validation_size} "
            "sequences reserved as a temporal validation split."
        ),
        (
            f"- Training config: Adam(lr={LEARNING_RATE}), MSE loss, epochs={EPOCHS}, "
            f"batch_size={BATCH_SIZE}, EarlyStopping(patience={EARLY_STOPPING_PATIENCE}, "
            "restore_best_weights=True)."
        ),
        "- Models evaluated: CNN-LSTM, Attention LSTM with custom dot-product attention, and a 2-head Temporal Transformer.",
        "",
        "### Architecture Definitions",
        "- CNN-LSTM: `Conv1D(64, kernel_size=3, activation='relu') -> MaxPooling1D(2) -> LSTM(64) -> Dense(1)`",
        "- Attention LSTM: `LSTM(64, return_sequences=True) -> DotProductAttention -> GlobalAveragePooling1D -> Dense(1)`",
        "- Temporal Transformer: `MultiHeadAttention(num_heads=2, key_dim=32) -> Add + LayerNormalization -> Dense(64, relu) -> GlobalAveragePooling1D -> Dense(1)`",
        "",
        "### Test Metrics",
        format_metrics_table(results_frame),
        "",
        "### Findings",
        (
            f"- Best advanced DL model by RMSE: `{best_row.model_name}` with "
            f"RMSE={best_row.RMSE:.4f}, MAE={best_row.MAE:.4f}, and R^2={best_row.R2:.4f}."
        ),
        (
            f"- Weakest advanced DL model by RMSE: `{worst_row.model_name}` with "
            f"RMSE={worst_row.RMSE:.4f}."
        ),
        "- All three models were trained on the same MinMax-scaled multivariate feature windows, so the differences reflect architecture choice rather than data leakage or split changes.",
        "- Prediction CSVs were written to `results/predictions/dl_advanced/`, architecture summaries were written to `results/*.txt`, and metrics were logged through `utils/metrics_tracker.py`.",
        "",
        "### Saved Artifacts",
        "- `src/models/dl_advanced_models.py`",
        "- `results/predictions/dl_advanced/*.csv`",
        "- `results/*_architecture.txt`",
        "- `results/metrics_registry.csv`",
    ]
    return "\n".join(lines)


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


def train_single_model(
    spec: ModelSpec,
    payload: dict[str, Any],
) -> ModelResult:
    from utils.metrics_tracker import compute_metrics, log_result

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = spec.factory((LOOKBACK_WINDOW, payload["X_train"].shape[-1]))
    architecture_path = save_architecture_summary(model, spec)

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
    predicted_actual = inverse_transform_target(
        predicted_scaled,
        payload["target_scaler"],
    )
    predictions_path = save_predictions(
        slug=spec.slug,
        dates=payload["test_index"],
        actual=payload["test_actual"],
        predicted=predicted_actual,
    )
    metrics = compute_metrics(payload["test_actual"], predicted_actual)
    log_result("dl_advanced", spec.model_name, metrics, predictions_path)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    return ModelResult(
        model_name=spec.model_name,
        slug=spec.slug,
        metrics=metrics,
        predictions_path=predictions_path,
        architecture_path=architecture_path,
        history=history_dict,
    )


def prepare_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    X_train_full, y_train_full = build_feature_sequences(
        raw_payload["train_feature_scaled"],
        raw_payload["train_target_scaled"],
        LOOKBACK_WINDOW,
    )
    X_test, y_test_scaled = build_test_sequences(
        raw_payload["train_feature_scaled"],
        raw_payload["test_feature_scaled"],
        raw_payload["train_target_scaled"],
        raw_payload["test_target_scaled"],
        LOOKBACK_WINDOW,
    )
    X_train, X_val, y_train, y_val = temporal_train_validation_split(
        X_train_full,
        y_train_full,
    )

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


def run_all_advanced_dl_models() -> pd.DataFrame:
    ensure_dirs()
    payload = prepare_payload(load_inputs())

    results: list[ModelResult] = []
    for spec in get_model_specs():
        result = train_single_model(spec, payload)
        results.append(result)

    results_frame = build_results_frame(results)
    upsert_research_log(build_research_log_section(results_frame, payload))
    return results_frame


def main() -> None:
    summary = run_all_advanced_dl_models()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

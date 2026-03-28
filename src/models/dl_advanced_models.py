"""
Advanced DL Models — Liquidity Index Forecasting
=================================================
Models:
  - CNN-LSTM: 1D convolutional feature extractor feeding LSTM
  - CNN-GRU: 1D convolutional feature extractor feeding GRU
  - Attention LSTM: LSTM with additive (Bahdanau) self-attention
  - Temporal Fusion Transformer (simplified): Multi-head self-attention
    with positional encoding and feed-forward layers
  - WaveNet-style Dilated Causal CNN
  - TCN (Temporal Convolutional Network)
Strategy: Same sequence-to-one setup as dl_rnn_models.py (seq_len=30).
          Same MinMaxScaler features, same train/val/test split.
Outputs:  results/dl_advanced/  |  plots/dl_advanced/
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "dl_advanced"
PLOTS_DIR = PROJECT_ROOT / "plots" / "dl_advanced"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 20
MAPE_EPSILON = 1e-6
LOG_SECTION_HEADER = "## Advanced DL Models"
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


@dataclass
class AdvDLResult:
    name: str
    slug: str
    predictions: np.ndarray
    metrics: dict[str, float]
    history: dict[str, list[float]]
    config: dict[str, Any]
    notes: str = ""


def load_artifacts() -> dict[str, Any]:
    arrays = joblib.load(ARTIFACTS_DIR / "preprocessed_arrays.joblib")
    dl_scalers = joblib.load(ARTIFACTS_DIR / "minmax_scalers.joblib")
    return {**arrays, "dl_scalers": dl_scalers}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


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


def inverse_transform_target(values: np.ndarray, dl_scalers: Any) -> np.ndarray:
    return dl_scalers["target"].inverse_transform(values.reshape(-1, 1)).ravel()


# ─── Custom Keras Layers ─────────────────────────────────────────────────────

class BahdanauAttention(layers.Layer):
    """Additive attention over LSTM hidden states."""
    def __init__(self, units: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, seq_len: int, d_model: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.encoding = tf.cast(angles[np.newaxis, :, :], tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.encoding[:, :tf.shape(x)[1], :]


# ─── Model Builders ──────────────────────────────────────────────────────────

def build_cnn_lstm(input_shape: tuple[int, int]) -> keras.Model:
    """1D CNN feature extractor → LSTM."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", dilation_rate=2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, dropout=0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_cnn_gru(input_shape: tuple[int, int]) -> keras.Model:
    """1D CNN feature extractor → GRU."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", dilation_rate=2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GRU(64, dropout=0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_attention_lstm(input_shape: tuple[int, int]) -> keras.Model:
    """LSTM with Bahdanau self-attention over all hidden states."""
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = BahdanauAttention(units=32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_attention_bilstm(input_shape: tuple[int, int]) -> keras.Model:
    """BiLSTM + self-attention."""
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inp)
    x = BahdanauAttention(units=64)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_transformer(input_shape: tuple[int, int], d_model: int = 64, num_heads: int = 4, ff_dim: int = 128) -> keras.Model:
    """Simplified Temporal Transformer: projection → positional encoding → multi-head attention → FFN → output."""
    inp = layers.Input(shape=input_shape)
    # Project features to d_model
    x = layers.Dense(d_model)(inp)
    x = PositionalEncoding(seq_len=input_shape[0], d_model=d_model)(x)

    # Transformer encoder block × 2
    for _ in range(2):
        # Multi-head self-attention
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x = layers.LayerNormalization()(x + attn_out)
        # Feed-forward
        ffn = layers.Dense(ff_dim, activation="gelu")(x)
        ffn = layers.Dense(d_model)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.LayerNormalization()(x + ffn)

    # Pool over time and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


def build_wavenet_cnn(input_shape: tuple[int, int]) -> keras.Model:
    """WaveNet-style dilated causal convolutions."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu")(inp)

    skip_connections = []
    for dilation in [1, 2, 4, 8, 16]:
        residual = x
        x = layers.Conv1D(64, kernel_size=2, padding="causal", dilation_rate=dilation, activation="tanh")(x)
        gate = layers.Conv1D(64, kernel_size=2, padding="causal", dilation_rate=dilation, activation="sigmoid")(residual)
        x = layers.Multiply()([x, gate])
        x = layers.Conv1D(64, kernel_size=1)(x)
        skip_connections.append(x)
        x = layers.Add()([x, residual]) if residual.shape[-1] == x.shape[-1] else x

    x = layers.Add()(skip_connections)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_tcn(input_shape: tuple[int, int]) -> keras.Model:
    """Temporal Convolutional Network with residual connections."""
    inp = layers.Input(shape=input_shape)
    x = inp
    n_filters = 64

    for dilation in [1, 2, 4, 8]:
        res = x
        x = layers.Conv1D(n_filters, kernel_size=3, padding="causal", dilation_rate=dilation)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(n_filters, kernel_size=3, padding="causal", dilation_rate=dilation)(x)
        x = layers.LayerNormalization()(x)
        # Residual connection (1x1 conv if dimensions mismatch)
        if res.shape[-1] != n_filters:
            res = layers.Conv1D(n_filters, kernel_size=1)(res)
        x = layers.Add()([x, res])
        x = layers.Activation("relu")(x)

    x = x[:, -1, :]  # Take last timestep
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_cnn_transformer(input_shape: tuple[int, int]) -> keras.Model:
    """CNN local feature extraction + Transformer global context."""
    inp = layers.Input(shape=input_shape)
    # Local CNN features
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)

    # Transformer on CNN features
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(64)(ffn)
    x = layers.LayerNormalization()(x + ffn)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


def get_model_specs(input_shape: tuple[int, int]) -> list[dict[str, Any]]:
    return [
        {
            "name": "CNN-LSTM",
            "slug": "cnn_lstm",
            "build_fn": lambda: build_cnn_lstm(input_shape),
            "notes": "1D causal CNN (dilated) → LSTM",
        },
        {
            "name": "CNN-GRU",
            "slug": "cnn_gru",
            "build_fn": lambda: build_cnn_gru(input_shape),
            "notes": "1D causal CNN (dilated) → GRU",
        },
        {
            "name": "Attention LSTM",
            "slug": "attention_lstm",
            "build_fn": lambda: build_attention_lstm(input_shape),
            "notes": "LSTM + Bahdanau additive attention",
        },
        {
            "name": "Attention BiLSTM",
            "slug": "attention_bilstm",
            "build_fn": lambda: build_attention_bilstm(input_shape),
            "notes": "Bidirectional LSTM + Bahdanau attention",
        },
        {
            "name": "Transformer (4-head)",
            "slug": "transformer_4h",
            "build_fn": lambda: build_transformer(input_shape, d_model=64, num_heads=4, ff_dim=128),
            "notes": "Simplified temporal transformer, 2 encoder blocks",
        },
        {
            "name": "WaveNet Dilated CNN",
            "slug": "wavenet_cnn",
            "build_fn": lambda: build_wavenet_cnn(input_shape),
            "notes": "Gated dilated causal convolutions (WaveNet-style)",
        },
        {
            "name": "TCN (Temporal Conv Net)",
            "slug": "tcn",
            "build_fn": lambda: build_tcn(input_shape),
            "notes": "Residual dilated causal TCN",
        },
        {
            "name": "CNN + Transformer",
            "slug": "cnn_transformer",
            "build_fn": lambda: build_cnn_transformer(input_shape),
            "notes": "CNN local features + Transformer global attention",
        },
    ]


def train_model(
    spec: dict[str, Any],
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_test_seq: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    keras.backend.clear_session()
    model = spec["build_fn"]()

    n_val = int(len(X_train_seq) * 0.1)
    X_val, y_val = X_train_seq[-n_val:], y_train_seq[-n_val:]
    X_tr, y_tr = X_train_seq[:-n_val], y_train_seq[:-n_val]

    cb_list = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=0,
        shuffle=False,
    )
    preds_scaled = model.predict(X_test_seq, verbose=0).ravel()
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return preds_scaled, history


def render_training_curve(result: AdvDLResult) -> None:
    h = result.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.get("loss", []), label="train")
    axes[0].plot(h.get("val_loss", []), label="val")
    axes[0].set_title(f"{result.name} — Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(h.get("mae", []), label="train")
    axes[1].plot(h.get("val_mae", []), label="val")
    axes[1].set_title(f"{result.name} — MAE")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{result.slug}_training.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_forecast_plot(test_dates: pd.DatetimeIndex, y_test: np.ndarray, result: AdvDLResult) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(test_dates, y_test, color="#2ca02c", lw=1.5, label="Actual")
    axes[0].plot(test_dates, result.predictions, color="#d62728", lw=1.5, ls="--", label="Predicted")
    axes[0].set_title(f"{result.name} — Test Forecast", fontsize=13)
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    m = result.metrics
    axes[0].set_xlabel(f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  DA={m['directional_accuracy']:.1f}%")
    residuals = y_test - result.predictions
    axes[1].bar(test_dates, residuals, color=["#d62728" if r < 0 else "#2ca02c" for r in residuals], width=2, alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Residuals"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{result.slug}_forecast.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_comparison_plot(results: list[AdvDLResult], y_test: np.ndarray, test_dates: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(test_dates, y_test, color="black", lw=2, label="Actual", zorder=5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for res, color in zip(results, colors):
        ax.plot(test_dates, res.predictions, lw=1, ls="--", color=color,
                label=f"{res.name} (MAE={res.metrics['mae']:.4f})", alpha=0.85)
    ax.set_title("Advanced DL Models — All Predictions vs Actual", fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="upper left"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "adv_dl_comparison_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_leaderboard(results: list[AdvDLResult]) -> pd.DataFrame:
    rows = [{"model": r.name, "slug": r.slug, **r.metrics} for r in results]
    df = pd.DataFrame(rows).sort_values("mae")
    df.to_csv(RESULTS_DIR / "adv_dl_leaderboard.csv", index=False)
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
    import re
    log = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else ""
    if LOG_SECTION_HEADER in log:
        before, _, after = log.partition(LOG_SECTION_HEADER)
        after_trimmed = re.sub(r".*?(?=\n## |\Z)", "", after, count=1, flags=re.DOTALL)
        content = before.rstrip() + "\n\n" + section + "\n" + after_trimmed
    else:
        content = log.rstrip() + "\n\n" + section + "\n"
    RESEARCH_LOG_PATH.write_text(content, encoding="utf-8")


def build_research_log_section(df: pd.DataFrame) -> str:
    best = df.iloc[0]
    section_lines = [
        LOG_SECTION_HEADER,
        "",
        "### Experimental Setup",
        f"- Sequence length: {SEQUENCE_LENGTH} timesteps",
        "- Same MinMaxScaler features as RNN models for direct comparability",
        "- CNN-LSTM/GRU: causal dilated convolutions extract local patterns, recurrent layers capture temporal dependencies",
        "- Attention models: LSTM/BiLSTM hidden states weighted by additive attention for interpretable focus",
        "- Transformer: sinusoidal positional encoding + 2 multi-head self-attention encoder blocks",
        "- WaveNet: gated dilated causal conv (dilations 1,2,4,8,16) with skip connections",
        "- TCN: residual dilated causal conv blocks (dilations 1,2,4,8)",
        "- CNN+Transformer: hybrid local-global architecture",
        "",
        "### Results Leaderboard",
        "",
        format_leaderboard_md(df),
        "",
        "### Researcher Notes",
        f"- Best advanced DL model: **{best.model}** — MAE={best.mae:.4f}, RMSE={best.rmse:.4f}, R²={best.r2:.4f}",
        "- Attention mechanisms allow inspection of which timesteps matter most for forecasting.",
        "- Transformer may under-perform on small datasets compared to recurrent models due to sample efficiency.",
        "- WaveNet/TCN offer strong inductive bias for sequential data without recurrence.",
        "",
        "### Saved Artifacts",
        "- `results/dl_advanced/<slug>_predictions.csv`",
        "- `results/dl_advanced/adv_dl_leaderboard.csv`",
        "- `plots/dl_advanced/<slug>_forecast.png`",
        "- `plots/dl_advanced/adv_dl_comparison_all.png`",
    ]
    return "\n".join(section_lines)


def run_all_advanced_dl_models() -> pd.DataFrame:
    print("=" * 60)
    print("ADVANCED DL MODELS ABLATION STUDY")
    print("=" * 60)
    ensure_dirs()
    data = load_artifacts()

    dl_scalers = data["dl_scalers"]
    X_train_dl = data["X_train_dl_scaled"].astype(np.float32)
    X_test_dl = data["X_test_dl_scaled"].astype(np.float32)
    y_train_dl = data["y_train_dl_scaled"].astype(np.float32)
    y_test_raw = data["y_test"].astype(np.float32)
    y_train_raw = data["y_train"].astype(np.float32)
    test_index = data["test_index"]

    X_train_seq, y_train_seq = build_sequences(X_train_dl, y_train_dl, SEQUENCE_LENGTH)

    combined_X = np.vstack([X_train_dl, X_test_dl])
    combined_y_raw = np.concatenate([y_train_raw, y_test_raw])
    combined_y_dl = np.concatenate([y_train_dl, data["y_test_dl_scaled"].astype(np.float32)])
    test_start_idx = len(X_train_dl)
    X_test_seq = np.array([
        combined_X[i - SEQUENCE_LENGTH:i]
        for i in range(test_start_idx, len(combined_X))
    ], dtype=np.float32)
    y_test_raw_seq = combined_y_raw[test_start_idx:]

    test_dates = pd.to_datetime(test_index)
    y_train_last = float(y_train_raw[-1])
    prev_actual = np.concatenate(([y_train_last], y_test_raw_seq[:-1]))

    n_features = X_train_dl.shape[1]
    input_shape = (SEQUENCE_LENGTH, n_features)
    specs = get_model_specs(input_shape)
    results: list[AdvDLResult] = []

    for spec in specs:
        print(f"\n  Training: {spec['name']}")
        preds_scaled, history = train_model(spec, X_train_seq, y_train_seq, X_test_seq)
        preds_raw = inverse_transform_target(preds_scaled, dl_scalers)
        metrics = compute_metrics(y_test_raw_seq, preds_raw, prev_actual)

        result = AdvDLResult(
            name=spec["name"],
            slug=spec["slug"],
            predictions=preds_raw,
            metrics=metrics,
            history=history,
            config={"seq_len": SEQUENCE_LENGTH, "input_shape": list(input_shape)},
            notes=spec["notes"],
        )
        results.append(result)

        df_pred = pd.DataFrame({
            "date": test_dates,
            "actual": y_test_raw_seq,
            "predicted": preds_raw,
            "error": y_test_raw_seq - preds_raw,
        })
        df_pred.to_csv(RESULTS_DIR / f"{spec['slug']}_predictions.csv", index=False)

        render_training_curve(result)
        render_forecast_plot(test_dates, y_test_raw_seq, result)

        print(f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}  DA={metrics['directional_accuracy']:.1f}%  Epochs={len(history.get('loss', []))}")

    metrics_all = {r.slug: r.metrics for r in results}
    (RESULTS_DIR / "adv_dl_all_metrics.json").write_text(json.dumps(metrics_all, indent=2), encoding="utf-8")

    leaderboard = save_leaderboard(results)
    render_comparison_plot(results, y_test_raw_seq, test_dates)
    section = build_research_log_section(leaderboard)
    upsert_research_log_section(section)

    print("\n" + "=" * 60)
    print("ADVANCED DL LEADERBOARD (sorted by MAE)")
    print("=" * 60)
    print(leaderboard[["model", "mae", "rmse", "r2", "directional_accuracy"]].to_string(index=False))
    print(f"\nArtifacts saved to: {RESULTS_DIR}")
    return leaderboard


if __name__ == "__main__":
    run_all_advanced_dl_models()

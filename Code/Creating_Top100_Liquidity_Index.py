"""
Create a composite liquidity index using the top 100 most liquid stocks.

Outputs:
1) Individual index plot: top100_liquidity_index_plot.png
2) Combined plot with NIFTY50: top100_vs_nifty50_overlay.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROXY_COLUMNS = ["ILLIQ_t", "DEPTH_t", "IMMED_t", "REV_t"]


def _winsorize_columns(
    df: pd.DataFrame,
    columns: list[str],
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    """Clip each proxy to global quantile bounds to limit outliers."""
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("Winsorization quantiles must satisfy 0 <= lower_q < upper_q <= 1")

    work_df = df.copy()
    for col in columns:
        q_low = work_df[col].quantile(lower_q)
        q_high = work_df[col].quantile(upper_q)
        if pd.notna(q_low) and pd.notna(q_high):
            work_df[col] = work_df[col].clip(lower=q_low, upper=q_high)
    return work_df


def select_top_100_liquid_stocks(
    proxy_df: pd.DataFrame,
    min_obs: int = 250,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Select top-N most liquid stocks using lowest average Amihud illiquidity.

    min_obs filters out symbols with insufficient history.
    """
    work_df = proxy_df.copy()
    work_df["DATE"] = pd.to_datetime(work_df["DATE"], errors="coerce")
    work_df["ILLIQ_t"] = pd.to_numeric(work_df["ILLIQ_t"], errors="coerce")
    work_df = work_df.dropna(subset=["DATE", "SYMBOL", "ILLIQ_t"])

    liquidity_rank = (
        work_df.groupby("SYMBOL", as_index=False)
        .agg(
            avg_illiq=("ILLIQ_t", "mean"),
            median_illiq=("ILLIQ_t", "median"),
            obs_count=("ILLIQ_t", "count"),
        )
    )

    eligible = liquidity_rank[liquidity_rank["obs_count"] >= min_obs].copy()
    eligible = eligible.sort_values(["avg_illiq", "median_illiq", "obs_count"], ascending=[True, True, False])

    selected = eligible.head(top_n).reset_index(drop=True)
    return selected


def construct_top100_liquidity_index(
    proxy_df: pd.DataFrame,
    selected_symbols: list[str],
    apply_winsorization: bool = True,
    winsor_lower_q: float = 0.01,
    winsor_upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct Top100 liquidity index via global PCA, matching market method style.

    Returns:
    - index_df: DATE, Top100_Liquidity_Index
    - weights_df: proxy loadings and contributions
    """
    required = ["DATE", "SYMBOL"] + PROXY_COLUMNS
    missing = [c for c in required if c not in proxy_df.columns]
    if missing:
        raise ValueError(f"Missing required proxy columns: {missing}")

    work_df = proxy_df[required].copy()
    work_df = work_df[work_df["SYMBOL"].isin(selected_symbols)].copy()
    work_df["DATE"] = pd.to_datetime(work_df["DATE"], errors="coerce")
    work_df[PROXY_COLUMNS] = work_df[PROXY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    work_df = work_df.dropna(subset=["DATE", "SYMBOL"] + PROXY_COLUMNS)
    work_df = work_df.sort_values(["DATE", "SYMBOL"]).reset_index(drop=True)

    if work_df.empty:
        raise ValueError("No usable rows found for selected symbols after dropping missing proxy values.")

    # IMMED_t is liquidity-oriented; flip sign so all proxies align with illiquidity.
    work_df["IMMED_t"] = -work_df["IMMED_t"]

    if apply_winsorization:
        work_df = _winsorize_columns(work_df, PROXY_COLUMNS, winsor_lower_q, winsor_upper_q)

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_std = scaler.fit_transform(work_df[PROXY_COLUMNS].to_numpy(dtype=float))
    x_std_df = pd.DataFrame(x_std, columns=PROXY_COLUMNS, index=work_df.index)

    pca = PCA(n_components=1)
    _ = pca.fit_transform(x_std)
    loadings = pca.components_[0].astype(float)

    stock_scores_df = work_df[["DATE", "SYMBOL", "ILLIQ_t"]].copy()
    stock_scores_df["Liquidity_Score"] = x_std_df.to_numpy(dtype=float).dot(loadings)

    index_df = (
        stock_scores_df.groupby("DATE", as_index=False)["Liquidity_Score"]
        .mean()
        .rename(columns={"Liquidity_Score": "Top100_Liquidity_Index_Raw"})
    )

    daily_illiq = (
        stock_scores_df.groupby("DATE", as_index=False)["ILLIQ_t"]
        .mean()
        .rename(columns={"ILLIQ_t": "ILLIQ_daily_mean"})
    )
    align_df = index_df.merge(daily_illiq, on="DATE", how="inner")
    idx_corr = align_df["Top100_Liquidity_Index_Raw"].corr(align_df["ILLIQ_daily_mean"])
    if pd.notna(idx_corr) and idx_corr > 0:
        index_df["Top100_Liquidity_Index_Raw"] = -index_df["Top100_Liquidity_Index_Raw"]
        loadings = -loadings

    mean_raw = index_df["Top100_Liquidity_Index_Raw"].mean()
    std_raw = index_df["Top100_Liquidity_Index_Raw"].std(ddof=0)
    if pd.notna(std_raw) and std_raw > 0:
        index_df["Top100_Liquidity_Index"] = (
            index_df["Top100_Liquidity_Index_Raw"] - mean_raw
        ) / std_raw
    else:
        index_df["Top100_Liquidity_Index"] = np.nan

    weights_abs = np.abs(loadings)
    abs_sum = float(weights_abs.sum())
    if abs_sum > 0:
        contrib = (weights_abs / abs_sum) * 100
    else:
        contrib = np.full_like(weights_abs, np.nan)

    weights_df = pd.DataFrame(
        {
            "Proxy": PROXY_COLUMNS,
            "PCA_Loading": loadings,
            "Absolute_Loading": weights_abs,
            "Contribution_Percent": contrib,
        }
    )

    return index_df[["DATE", "Top100_Liquidity_Index"]].copy(), weights_df


def plot_top100_index(index_df: pd.DataFrame, output_path: Path) -> None:
    """Plot individual Top100 liquidity index time series."""
    plot_df = index_df.copy()
    plot_df["DATE"] = pd.to_datetime(plot_df["DATE"], errors="coerce")
    plot_df = plot_df.dropna(subset=["DATE", "Top100_Liquidity_Index"]).sort_values("DATE")

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["DATE"], plot_df["Top100_Liquidity_Index"], linewidth=1.2, color="tab:green")
    plt.title("Top 100 Most Liquid Stocks: Composite Liquidity Index")
    plt.xlabel("DATE")
    plt.ylabel("Top100 Liquidity Index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top100_vs_nifty50(
    top100_index_df: pd.DataFrame,
    nifty50_index_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Create combined normalized overlay of Top100 and NIFTY50 indices."""
    lhs = top100_index_df.copy()
    rhs = nifty50_index_df.copy()

    lhs["DATE"] = pd.to_datetime(lhs["DATE"], errors="coerce")
    rhs["DATE"] = pd.to_datetime(rhs["DATE"], errors="coerce")

    merged = pd.merge(lhs, rhs, on="DATE", how="inner")
    merged = merged.dropna(subset=["Top100_Liquidity_Index", "NIFTY50_Liquidity_Index"])

    top_std = merged["Top100_Liquidity_Index"].std(ddof=0)
    nifty_std = merged["NIFTY50_Liquidity_Index"].std(ddof=0)

    if pd.notna(top_std) and top_std > 0:
        merged["Top100_Normalized"] = (
            merged["Top100_Liquidity_Index"] - merged["Top100_Liquidity_Index"].mean()
        ) / top_std
    else:
        merged["Top100_Normalized"] = np.nan

    if pd.notna(nifty_std) and nifty_std > 0:
        merged["NIFTY50_Normalized"] = (
            merged["NIFTY50_Liquidity_Index"] - merged["NIFTY50_Liquidity_Index"].mean()
        ) / nifty_std
    else:
        merged["NIFTY50_Normalized"] = np.nan

    plt.figure(figsize=(14, 6))
    plt.plot(merged["DATE"], merged["Top100_Normalized"], label="Top100 Index (Normalized)", linewidth=1.5)
    plt.plot(merged["DATE"], merged["NIFTY50_Normalized"], label="NIFTY50 Index (Normalized)", linewidth=1.5)
    plt.title("Normalized Liquidity Index Comparison: Top100 vs NIFTY50", fontsize=13, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Normalized Index")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return merged


def _zscore_series(series: pd.Series) -> pd.Series:
    """Return z-scored series; if std is zero/NaN, return NaNs."""
    std_val = series.std(ddof=0)
    if pd.notna(std_val) and std_val > 0:
        return (series - series.mean()) / std_val
    return pd.Series(np.nan, index=series.index)


def plot_all_three_and_pairs(
    market_index_df: pd.DataFrame,
    nifty50_index_df: pd.DataFrame,
    top100_index_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create one 6-panel image: 3 individual series + 3 pairwise overlays."""
    market = market_index_df[["DATE", "Market_Liquidity_Index"]].copy()
    nifty = nifty50_index_df[["DATE", "NIFTY50_Liquidity_Index"]].copy()
    top = top100_index_df[["DATE", "Top100_Liquidity_Index"]].copy()

    for df in [market, nifty, top]:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df.dropna(subset=["DATE"], inplace=True)
        df.sort_values("DATE", inplace=True)

    market["Market_Z"] = _zscore_series(market["Market_Liquidity_Index"])
    nifty["NIFTY50_Z"] = _zscore_series(nifty["NIFTY50_Liquidity_Index"])
    top["Top100_Z"] = _zscore_series(top["Top100_Liquidity_Index"])

    mn = pd.merge(market[["DATE", "Market_Z"]], nifty[["DATE", "NIFTY50_Z"]], on="DATE", how="inner")
    mt = pd.merge(market[["DATE", "Market_Z"]], top[["DATE", "Top100_Z"]], on="DATE", how="inner")
    nt = pd.merge(nifty[["DATE", "NIFTY50_Z"]], top[["DATE", "Top100_Z"]], on="DATE", how="inner")

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=False)

    # Individual panels
    axes[0, 0].plot(market["DATE"], market["Market_Z"], color="tab:blue", linewidth=1.1)
    axes[0, 0].set_title("Market Liquidity Index (Normalized)")
    axes[0, 0].set_ylabel("Z-score")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(nifty["DATE"], nifty["NIFTY50_Z"], color="tab:orange", linewidth=1.1)
    axes[0, 1].set_title("NIFTY50 Liquidity Index (Normalized)")
    axes[0, 1].set_ylabel("Z-score")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(top["DATE"], top["Top100_Z"], color="tab:green", linewidth=1.1)
    axes[1, 0].set_title("Top100 Liquidity Index (Normalized)")
    axes[1, 0].set_ylabel("Z-score")
    axes[1, 0].grid(alpha=0.3)

    # Pairwise panels
    axes[1, 1].plot(mn["DATE"], mn["Market_Z"], label="Market", color="tab:blue", linewidth=1.1)
    axes[1, 1].plot(mn["DATE"], mn["NIFTY50_Z"], label="NIFTY50", color="tab:orange", linewidth=1.1)
    axes[1, 1].set_title("Market vs NIFTY50")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylabel("Z-score")
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(mt["DATE"], mt["Market_Z"], label="Market", color="tab:blue", linewidth=1.1)
    axes[2, 0].plot(mt["DATE"], mt["Top100_Z"], label="Top100", color="tab:green", linewidth=1.1)
    axes[2, 0].set_title("Market vs Top100")
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].set_xlabel("Date")
    axes[2, 0].set_ylabel("Z-score")
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(nt["DATE"], nt["NIFTY50_Z"], label="NIFTY50", color="tab:orange", linewidth=1.1)
    axes[2, 1].plot(nt["DATE"], nt["Top100_Z"], label="Top100", color="tab:green", linewidth=1.1)
    axes[2, 1].set_title("NIFTY50 vs Top100")
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].set_xlabel("Date")
    axes[2, 1].set_ylabel("Z-score")
    axes[2, 1].grid(alpha=0.3)

    # Keep x-labels clean for upper rows
    axes[0, 0].set_xlabel("Date")
    axes[0, 1].set_xlabel("Date")
    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")

    plt.suptitle("Liquidity Indices: Individuals and Pairwise Comparisons", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    plots_dir = code_dir / "plots"

    proxy_input_path = code_dir / "daily_liquidity_proxies.csv"
    nifty50_index_path = code_dir / "nifty50_liquidity_index.csv"
    market_index_path = code_dir / "market_liquidity_index.csv"

    selected_stocks_path = code_dir / "top100_most_liquid_stocks.csv"
    top100_index_path = code_dir / "top100_liquidity_index.csv"
    top100_weights_path = code_dir / "top100_proxy_pca_weights.csv"
    compare_csv_path = code_dir / "top100_vs_nifty50_comparison.csv"

    top100_plot_path = plots_dir / "top100_liquidity_index_plot.png"
    combined_plot_path = plots_dir / "top100_vs_nifty50_overlay.png"
    six_panel_plot_path = plots_dir / "all_indices_six_panel.png"

    if not proxy_input_path.exists():
        raise FileNotFoundError(f"Input proxy file not found: {proxy_input_path}")
    if not nifty50_index_path.exists():
        raise FileNotFoundError(f"NIFTY50 index file not found: {nifty50_index_path}")
    if not market_index_path.exists():
        raise FileNotFoundError(f"Market index file not found: {market_index_path}")

    print("=" * 72)
    print("Top 100 Most Liquid Stocks: Composite Liquidity Index")
    print("=" * 72)

    print(f"Loading proxy data: {proxy_input_path}")
    proxy_df = pd.read_csv(proxy_input_path)

    selected_df = select_top_100_liquid_stocks(proxy_df, min_obs=250, top_n=100)
    if selected_df.empty:
        raise ValueError("No stocks selected for Top100. Try lowering min_obs.")

    selected_df.to_csv(selected_stocks_path, index=False)
    selected_symbols = selected_df["SYMBOL"].tolist()
    print(f"Selected {len(selected_symbols)} stocks. Saved list: {selected_stocks_path}")

    top100_index_df, weights_df = construct_top100_liquidity_index(proxy_df, selected_symbols)
    top100_index_df.to_csv(top100_index_path, index=False)
    weights_df.to_csv(top100_weights_path, index=False)

    print(f"Saved Top100 index CSV: {top100_index_path}")
    print(f"Saved Top100 PCA weights: {top100_weights_path}")

    plot_top100_index(top100_index_df, top100_plot_path)
    print(f"Saved individual plot: {top100_plot_path}")

    nifty50_index_df = pd.read_csv(nifty50_index_path)
    comparison_df = plot_top100_vs_nifty50(top100_index_df, nifty50_index_df, combined_plot_path)
    comparison_df.to_csv(compare_csv_path, index=False)

    market_index_df = pd.read_csv(market_index_path)
    plot_all_three_and_pairs(
        market_index_df=market_index_df,
        nifty50_index_df=nifty50_index_df,
        top100_index_df=top100_index_df,
        output_path=six_panel_plot_path,
    )

    print(f"Saved combined plot: {combined_plot_path}")
    print(f"Saved 6-panel plot: {six_panel_plot_path}")
    print(f"Saved comparison CSV: {compare_csv_path}")

    print("\nOutput summary:")
    print(f"- Top100 stocks list: {selected_stocks_path}")
    print(f"- Top100 index: {top100_index_path}")
    print(f"- Top100 weights: {top100_weights_path}")
    print(f"- Plot 1 (individual): {top100_plot_path}")
    print(f"- Plot 2 (combined): {combined_plot_path}")
    print(f"- Plot 3 (all 6 panels): {six_panel_plot_path}")


if __name__ == "__main__":
    main()

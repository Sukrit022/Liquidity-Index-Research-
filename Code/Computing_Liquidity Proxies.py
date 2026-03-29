from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROXY_COLUMNS = ["ILLIQ_t", "DEPTH_t", "IMMED_t", "REV_t"]


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize common NSE column-name variants to a single schema."""
	rename_map = {
		"PREV. CLOSE": "PREV_CLOSE",
		"PREV CLOSE": "PREV_CLOSE",
		"NO OF TRADES": "NO_OF_TRADES",
		"NO. OF TRADES": "NO_OF_TRADES",
	}

	df = df.copy()
	df.columns = [str(col).strip().upper() for col in df.columns]
	df = df.rename(columns=rename_map)
	return df


def _winsorize_columns(
	df: pd.DataFrame,
	columns: list[str],
	lower_q: float,
	upper_q: float,
) -> pd.DataFrame:
	"""Clip each column to its global quantile bounds to limit outlier impact."""
	if not (0 <= lower_q < upper_q <= 1):
		raise ValueError("Winsorization quantiles must satisfy 0 <= lower_q < upper_q <= 1")

	work_df = df.copy()
	for col in columns:
		q_low = work_df[col].quantile(lower_q)
		q_high = work_df[col].quantile(upper_q)
		if pd.notna(q_low) and pd.notna(q_high):
			work_df[col] = work_df[col].clip(lower=q_low, upper=q_high)
	return work_df


def compute_daily_liquidity_proxies(df: pd.DataFrame, drop_na_rows: bool = False) -> pd.DataFrame:
	"""
	Compute daily liquidity proxies for each stock.

	Required columns (case-insensitive after normalization):
	DATE, CLOSE, PREV_CLOSE, VALUE, NO_OF_TRADES, SYMBOL

	Returns a DataFrame with:
	DATE, SYMBOL, ILLIQ_t, DEPTH_t, IMMED_t, REV_t
	"""
	df = _normalize_column_names(df)

	required_cols = ["DATE", "CLOSE", "PREV_CLOSE", "VALUE", "NO_OF_TRADES", "SYMBOL"]
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns: {missing_cols}")

	# Parse date and numeric fields, then sort for reliable time-series operations.
	df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
	numeric_cols = ["CLOSE", "PREV_CLOSE", "VALUE", "NO_OF_TRADES"]
	df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
	df = df.dropna(subset=["DATE", "SYMBOL"]).sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

	# Guard against invalid denominators for division and logarithms.
	prev_close_safe = df["PREV_CLOSE"].where(df["PREV_CLOSE"] > 0)
	value_safe = df["VALUE"].where(df["VALUE"] > 0)
	trades_safe = df["NO_OF_TRADES"].where(df["NO_OF_TRADES"] > 0)

	with np.errstate(divide="ignore", invalid="ignore"):
		# Daily return: R_t = (CLOSE - PREV_CLOSE) / PREV_CLOSE
		df["R_t"] = (df["CLOSE"] - prev_close_safe) / prev_close_safe

		# A. Price impact (Amihud illiquidity): |R_t| / VALUE
		df["ILLIQ_t"] = df["R_t"].abs() / value_safe

		# B. Depth: -log(VALUE)
		df["DEPTH_t"] = -np.log(value_safe)

		# C. Immediacy: log(NO_OF_TRADES / VALUE)
		df["IMMED_t"] = np.log(trades_safe / value_safe)

	# D. Resilience: -(R_t * R_(t-1))
	df["R_t_lag1"] = df.groupby("SYMBOL", sort=False)["R_t"].shift(1)
	df["REV_t"] = -(df["R_t"] * df["R_t_lag1"])

	result = df[["DATE", "SYMBOL", "ILLIQ_t", "DEPTH_t", "IMMED_t", "REV_t"]].copy()
	if drop_na_rows:
		result = result.dropna().reset_index(drop=True)

	return result


def analyze_proxy_correlations(
	df: pd.DataFrame,
	threshold: float = 0.8,
	heatmap_path: Path | None = None,
) -> dict[str, pd.DataFrame | dict[int, pd.DataFrame]]:
	"""
	Run correlation diagnostics for liquidity proxies.

	Steps:
	1) Select proxy columns and drop missing rows.
	2) Compute overall Pearson correlation matrix.
	3) Identify highly correlated pairs (|corr| > threshold).
	4) Compute year-wise correlation matrices.
	5) Optionally save correlation heatmap.
	"""
	required = ["DATE"] + PROXY_COLUMNS
	missing_cols = [col for col in required if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns for correlation analysis: {missing_cols}")

	work_df = df.copy()
	work_df["DATE"] = pd.to_datetime(work_df["DATE"], errors="coerce")
	work_df = work_df.dropna(subset=["DATE"])

	# Keep only proxies and remove rows with any missing proxy values.
	proxy_df = work_df[PROXY_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="any")

	# Overall Pearson correlation matrix.
	overall_corr = proxy_df.corr(method="pearson")

	# Find highly correlated proxy pairs without duplicates.
	high_corr_rows = []
	for i, col_i in enumerate(PROXY_COLUMNS):
		for j in range(i + 1, len(PROXY_COLUMNS)):
			col_j = PROXY_COLUMNS[j]
			corr_val = overall_corr.loc[col_i, col_j]
			if pd.notna(corr_val) and abs(corr_val) > threshold:
				high_corr_rows.append(
					{
						"proxy_1": col_i,
						"proxy_2": col_j,
						"correlation": float(corr_val),
						"abs_correlation": float(abs(corr_val)),
					}
				)
	high_corr_pairs = pd.DataFrame(
		high_corr_rows,
		columns=["proxy_1", "proxy_2", "correlation", "abs_correlation"],
	)
	if not high_corr_pairs.empty:
		high_corr_pairs = high_corr_pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

	# Year-wise correlation stability check.
	work_with_year = work_df[["DATE"] + PROXY_COLUMNS].copy()
	work_with_year[PROXY_COLUMNS] = work_with_year[PROXY_COLUMNS].apply(pd.to_numeric, errors="coerce")
	work_with_year = work_with_year.dropna(subset=PROXY_COLUMNS)
	work_with_year["year"] = work_with_year["DATE"].dt.year

	yearly_corr_dict: dict[int, pd.DataFrame] = {}
	yearly_corr_frames = []
	for year, group in work_with_year.groupby("year"):
		corr_mat = group[PROXY_COLUMNS].corr(method="pearson")
		yearly_corr_dict[int(year)] = corr_mat
		yearly_corr_frames.append(corr_mat.assign(year=int(year)).set_index("year", append=True))

	if yearly_corr_frames:
		yearly_corr_summary = pd.concat(yearly_corr_frames)
		yearly_corr_summary.index = yearly_corr_summary.index.set_names(["proxy", "year"])
	else:
		yearly_corr_summary = pd.DataFrame()

	# Heatmap visualization for the overall matrix.
	if heatmap_path is not None:
		plt.figure(figsize=(8, 6))
		sns.heatmap(
			overall_corr,
			annot=True,
			fmt=".2f",
			cmap="coolwarm",
			vmin=-1,
			vmax=1,
			square=True,
		)
		plt.title("Liquidity Proxies Correlation Heatmap")
		plt.tight_layout()
		heatmap_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(heatmap_path, dpi=150)
		plt.close()

	return {
		"overall_corr": overall_corr,
		"high_corr_pairs": high_corr_pairs,
		"yearly_corr_dict": yearly_corr_dict,
		"yearly_corr_summary": yearly_corr_summary,
	}


def construct_market_liquidity_index(
	df: pd.DataFrame,
	apply_winsorization: bool = True,
	winsor_lower_q: float = 0.01,
	winsor_upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Construct market liquidity index using a time-consistent (global) PCA.

	Workflow:
	1) Align proxy direction to illiquidity.
	2) Optionally winsorize proxies to reduce outlier-driven spikes.
	3) Standardize proxies globally over the full sample.
	4) Fit PCA once on full standardized proxy matrix.
	5) Use global PC1 loadings to compute stock-day liquidity scores.
	6) Average scores cross-sectionally by date to get market index.
	7) Adjust sign so higher index means higher liquidity.
	8) Add 30-day rolling mean and z-score normalization over time.
	"""
	required = ["DATE", "SYMBOL"] + PROXY_COLUMNS
	missing_cols = [col for col in required if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns for PCA index construction: {missing_cols}")

	# Step 1: Data preparation.
	work_df = df[required].copy()
	work_df["DATE"] = pd.to_datetime(work_df["DATE"], errors="coerce")
	work_df[PROXY_COLUMNS] = work_df[PROXY_COLUMNS].apply(pd.to_numeric, errors="coerce")
	work_df = work_df.dropna(subset=["DATE", "SYMBOL"] + PROXY_COLUMNS)
	work_df = work_df.sort_values(["DATE", "SYMBOL"]).reset_index(drop=True)

	# Step 2: Direction consistency for illiquidity interpretation.
	# IMMED_t = log(NO_OF_TRADES / VALUE) is typically liquidity-oriented,
	# so multiply by -1 so larger values imply higher illiquidity.
	work_df["IMMED_t"] = -work_df["IMMED_t"]

	# Step 3: Optional winsorization to control extreme outliers.
	if apply_winsorization:
		work_df = _winsorize_columns(
			df=work_df,
			columns=PROXY_COLUMNS,
			lower_q=winsor_lower_q,
			upper_q=winsor_upper_q,
		)

	# Step 4: Global standardization across full panel (time + cross-section).
	scaler = StandardScaler(with_mean=True, with_std=True)
	x_std = scaler.fit_transform(work_df[PROXY_COLUMNS].to_numpy(dtype=float))
	x_std_df = pd.DataFrame(x_std, columns=PROXY_COLUMNS, index=work_df.index)

	# Step 5: Global PCA fit and PC1 extraction.
	pca = PCA(n_components=1)
	pc1_scores = pca.fit_transform(x_std).ravel()
	loadings = pca.components_[0].astype(float)

	stock_scores_df = work_df[["DATE", "SYMBOL", "ILLIQ_t"]].copy()
	stock_scores_df["PC1_raw"] = pc1_scores

	# Step 6: Stock-level liquidity score as weighted sum of standardized proxies.
	# Equivalent to PC1 score up to scaling/sign.
	stock_scores_df["Liquidity_Score"] = x_std_df.to_numpy(dtype=float).dot(loadings)

	# Step 7: Daily market index via cross-sectional mean.
	market_index_df = (
		stock_scores_df.groupby("DATE", as_index=False)["Liquidity_Score"]
		.mean()
		.rename(columns={"Liquidity_Score": "Market_Liquidity_Index_Raw"})
	)

	# Step 8: Direction adjustment using relationship with daily Amihud average.
	# Keep final index liquidity-oriented: higher index => lower Amihud illiquidity.
	daily_illiq = (
		stock_scores_df.groupby("DATE", as_index=False)["ILLIQ_t"]
		.mean()
		.rename(columns={"ILLIQ_t": "ILLIQ_daily_mean"})
	)
	align_df = market_index_df.merge(daily_illiq, on="DATE", how="inner")
	idx_corr = align_df["Market_Liquidity_Index_Raw"].corr(align_df["ILLIQ_daily_mean"])
	if pd.notna(idx_corr) and idx_corr > 0:
		stock_scores_df["Liquidity_Score"] = -stock_scores_df["Liquidity_Score"]
		stock_scores_df["PC1_raw"] = -stock_scores_df["PC1_raw"]
		loadings = -loadings
		market_index_df["Market_Liquidity_Index_Raw"] = -market_index_df["Market_Liquidity_Index_Raw"]

	# Step 9: Final normalization over time (z-score).
	mean_raw = market_index_df["Market_Liquidity_Index_Raw"].mean()
	std_raw = market_index_df["Market_Liquidity_Index_Raw"].std(ddof=0)
	if pd.notna(std_raw) and std_raw > 0:
		market_index_df["Market_Liquidity_Index"] = (
			market_index_df["Market_Liquidity_Index_Raw"] - mean_raw
		) / std_raw
	else:
		market_index_df["Market_Liquidity_Index"] = np.nan

	weights_abs = np.abs(loadings)
	abs_sum = float(weights_abs.sum())
	if abs_sum > 0:
		contrib = (weights_abs / abs_sum) * 100
	else:
		contrib = np.full_like(weights_abs, np.nan)

	proxy_weights_df = pd.DataFrame(
		{
			"Proxy": PROXY_COLUMNS,
			"PCA_Loading": loadings,
			"Absolute_Loading": weights_abs,
			"Contribution_Percent": contrib,
		}
	)

	return stock_scores_df, market_index_df[["DATE", "Market_Liquidity_Index"]].copy(), proxy_weights_df


def plot_market_liquidity_index(market_index_df: pd.DataFrame, plot_path: Path) -> None:
	"""Plot market liquidity index time series and save image."""
	plot_df = market_index_df.copy()
	plot_df["DATE"] = pd.to_datetime(plot_df["DATE"], errors="coerce")
	plot_df = plot_df.dropna(subset=["DATE", "Market_Liquidity_Index"]).sort_values("DATE")

	plt.figure(figsize=(12, 5))
	plt.plot(plot_df["DATE"], plot_df["Market_Liquidity_Index"], linewidth=1.2)
	plt.title("Market Liquidity Index (2010-2025)")
	plt.xlabel("DATE")
	plt.ylabel("Market Liquidity Index")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(plot_path, dpi=150)
	plt.close()


def main() -> None:
	"""
	Example runner:
	- Reads a combined daily stock CSV.
	- Computes liquidity proxies.
	- Writes output CSV in the same folder as this script.
	"""
	code_dir = Path(__file__).resolve().parent
	plots_dir = code_dir / "plots"
	input_path = code_dir / "combined_nse_daily_data.csv"
	proxy_output_path = code_dir / "daily_liquidity_proxies.csv"
	heatmap_output_path = plots_dir / "proxy_correlation_heatmap.png"
	index_output_path = code_dir / "market_liquidity_index.csv"
	index_plot_output_path = plots_dir / "market_liquidity_index_plot.png"
	weights_output_path = code_dir / "proxy_pca_weights.csv"

	if not input_path.exists():
		print(f"Input file not found: {input_path}")
		print("Create this file with columns: DATE, CLOSE, PREV_CLOSE, VALUE, NO_OF_TRADES, SYMBOL")
		return

	raw_df = pd.read_csv(input_path)
	proxy_df = compute_daily_liquidity_proxies(raw_df, drop_na_rows=False)
	proxy_df.to_csv(proxy_output_path, index=False)

	analysis = analyze_proxy_correlations(
		df=proxy_df,
		threshold=0.8,
		heatmap_path=heatmap_output_path,
	)

	_, market_index_df, proxy_weights_df = construct_market_liquidity_index(proxy_df)
	market_index_df[["DATE", "Market_Liquidity_Index"]].to_csv(index_output_path, index=False)
	proxy_weights_df.to_csv(weights_output_path, index=False)
	plot_market_liquidity_index(market_index_df, index_plot_output_path)

	_ = analysis
	for stale_file in [
		code_dir / "proxy_correlation_matrix.csv",
		code_dir / "high_correlation_pairs.csv",
		code_dir / "yearly_proxy_correlations.csv",
		plots_dir / "market_liquidity_index_30d_plot.png",
	]:
		if stale_file.exists():
			stale_file.unlink()
	print(f"Saved liquidity proxies: {proxy_output_path}")
	print(f"Saved heatmap: {heatmap_output_path}")
	print(f"Saved market liquidity index CSV: {index_output_path}")
	print(f"Saved PCA proxy weights CSV: {weights_output_path}")
	print(f"Saved market liquidity index plot: {index_plot_output_path}")


if __name__ == "__main__":
	main()

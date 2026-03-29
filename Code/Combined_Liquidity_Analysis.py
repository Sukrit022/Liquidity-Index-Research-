"""
Combined Liquidity Analysis Script

Orchestrates three separate analyses in sequence:
1. Computing Liquidity Proxies - calculates daily proxies and market index
2. Creating NIFTY50 Composite Index - creates NIFTY50 benchmark index
3. Creating Top100 Liquidity Index - creates index for top 100 liquid stocks

No logic has been changed from the original files.
All computations and outputs remain identical.
"""

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
import pickle

# ============================================================================
# PART 1: COMPUTING LIQUIDITY PROXIES
# ============================================================================

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


def part1_computing_liquidity_proxies() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Part 1: Compute liquidity proxies and market index.
	Returns proxy_df and market_index_df for use in subsequent parts.
	"""
	print("\n" + "="*80)
	print("PART 1: COMPUTING LIQUIDITY PROXIES AND MARKET INDEX")
	print("="*80)
	
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
		return None, None

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
	
	return proxy_df, market_index_df


# ============================================================================
# PART 2: CREATING NIFTY50 COMPOSITE INDEX
# ============================================================================

def _winsorize_columns_nifty(
	df: pd.DataFrame,
	columns: list[str],
	lower_q: float,
	upper_q: float,
) -> pd.DataFrame:
	"""Clip each column to global quantile bounds to limit outlier impact."""
	if not (0 <= lower_q < upper_q <= 1):
		raise ValueError("Winsorization quantiles must satisfy 0 <= lower_q < upper_q <= 1")

	work_df = df.copy()
	for col in columns:
		q_low = work_df[col].quantile(lower_q)
		q_high = work_df[col].quantile(upper_q)
		if pd.notna(q_low) and pd.notna(q_high):
			work_df[col] = work_df[col].clip(lower=q_low, upper=q_high)
	return work_df


def load_and_prepare_nifty50(csv_path: str | Path) -> pd.DataFrame:
	"""Load NIFTY50.csv and prepare clean OHLCV data."""
	df = pd.read_csv(csv_path)
	df.columns = [str(col).strip().upper() for col in df.columns]

	# File has metadata rows (Ticker row and Date label row) before actual data.
	if df.shape[0] > 0 and isinstance(df.iloc[0, 0], str) and 'TICKER' in str(df.iloc[0, 0]).upper():
		df = df.iloc[2:].reset_index(drop=True)

	if 'PRICE' in df.columns:
		df = df.rename(columns={'PRICE': 'DATE'})

	required_cols = ['DATE', 'CLOSE', 'HIGH', 'LOW', 'VOLUME']
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required NIFTY50 columns: {missing_cols}")

	df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
	for col in ['CLOSE', 'HIGH', 'LOW', 'VOLUME']:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	df = df.dropna(subset=['DATE', 'CLOSE']).sort_values('DATE').reset_index(drop=True)
	df['SYMBOL'] = 'NIFTY50'
	return df


def compute_nifty50_liquidity_proxies(df: pd.DataFrame, drop_na_rows: bool = True) -> pd.DataFrame:
	"""Compute NIFTY50 liquidity proxies from OHLCV data."""
	work_df = df.copy()
	work_df['DATE'] = pd.to_datetime(work_df['DATE'], errors='coerce')
	work_df = work_df.sort_values('DATE').reset_index(drop=True)

	# Use traded value proxy because index file doesn't include explicit traded value.
	work_df['VALUE_PROXY'] = (work_df['CLOSE'] * work_df['VOLUME']) + 1
	work_df['R_t'] = work_df['CLOSE'].pct_change()

	with np.errstate(divide='ignore', invalid='ignore'):
		work_df['ILLIQ_t'] = work_df['R_t'].abs() / work_df['VALUE_PROXY']
		work_df['DEPTH_t'] = -np.log(work_df['VALUE_PROXY'])
		work_df['SPREAD_t'] = (work_df['HIGH'] - work_df['LOW']) / work_df['CLOSE']

	work_df['R_t_lag1'] = work_df['R_t'].shift(1)
	work_df['REV_t'] = -(work_df['R_t'] * work_df['R_t_lag1'])

	result_df = work_df[['DATE', 'SYMBOL', 'ILLIQ_t', 'DEPTH_t', 'SPREAD_t', 'REV_t']].copy()
	if drop_na_rows:
		result_df = result_df.dropna().reset_index(drop=True)
	return result_df


def create_nifty50_composite_index(
	df: pd.DataFrame,
	proxy_columns: list[str] | None = None,
	n_components: int = 1,
	output_dir: str | Path = '.',
	apply_winsorization: bool = True,
	winsor_lower_q: float = 0.01,
	winsor_upper_q: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
	"""
	Apply PCA to NIFTY50 proxies using a method consistent with market-index construction.
	
	Parameters:
	- df: DataFrame with DATE and proxy columns
	- proxy_columns: List of proxy column names to use in PCA
	- n_components: Number of principal components (default: 1 for single composite index)
	- output_dir: Directory to save PCA weights and scaler
	- apply_winsorization: Whether to winsorize proxies before PCA
	- winsor_lower_q/winsor_upper_q: Winsorization quantiles
	
	Returns:
	- DataFrame with DATE, NIFTY50_Liquidity_Index, NIFTY50_Liquidity_Index_30D
	- Dictionary with PCA details (model, scaler, explained variance)
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	if proxy_columns is None:
		# Use available proxy columns
		proxy_columns = ['ILLIQ_t', 'DEPTH_t', 'SPREAD_t', 'REV_t']
	
	# Filter to available columns
	available_proxies = [col for col in proxy_columns if col in df.columns]
	if not available_proxies:
		raise ValueError(f"No proxy columns found. Available: {df.columns.tolist()}")
	
	print(f"Using proxies for PCA: {available_proxies}")
	if n_components != 1:
		print("n_components != 1 requested; using PC1 for index output to preserve methodology consistency.")
	
	work_df = df.copy()
	work_df['DATE'] = pd.to_datetime(work_df['DATE'], errors='coerce')
	work_df[available_proxies] = work_df[available_proxies].apply(pd.to_numeric, errors='coerce')
	work_df = work_df.dropna(subset=['DATE'] + available_proxies)
	work_df = work_df.sort_values('DATE').reset_index(drop=True)
	print(f"Valid observations for PCA: {len(work_df)} / {len(df)}")

	# Direction consistency: if IMMED_t exists, invert to illiquidity orientation.
	if 'IMMED_t' in available_proxies:
		work_df['IMMED_t'] = -work_df['IMMED_t']

	# Optional winsorization to reduce outlier-driven spikes.
	if apply_winsorization:
		work_df = _winsorize_columns_nifty(
			df=work_df,
			columns=available_proxies,
			lower_q=winsor_lower_q,
			upper_q=winsor_upper_q,
		)

	# Global standardization across full sample.
	scaler = StandardScaler(with_mean=True, with_std=True)
	x_std = scaler.fit_transform(work_df[available_proxies].to_numpy(dtype=float))
	x_std_df = pd.DataFrame(x_std, columns=available_proxies, index=work_df.index)

	# Fit PCA once and use PC1 loadings.
	pca = PCA(n_components=1)
	_ = pca.fit_transform(x_std)
	loadings = pca.components_[0].astype(float)
	pc1_raw = x_std_df.to_numpy(dtype=float).dot(loadings)

	result_df = work_df[['DATE', 'ILLIQ_t']].copy() if 'ILLIQ_t' in work_df.columns else work_df[['DATE']].copy()
	result_df['NIFTY50_Liquidity_Index_Raw'] = pc1_raw

	# Align direction so final index is liquidity-oriented (higher => more liquid).
	if 'ILLIQ_t' in result_df.columns:
		idx_corr = result_df['NIFTY50_Liquidity_Index_Raw'].corr(result_df['ILLIQ_t'])
		if pd.notna(idx_corr) and idx_corr > 0:
			result_df['NIFTY50_Liquidity_Index_Raw'] = -result_df['NIFTY50_Liquidity_Index_Raw']
			loadings = -loadings

	# 30-day rolling mean (backward-looking only).
	result_df['NIFTY50_Liquidity_Index_30D'] = (
		result_df['NIFTY50_Liquidity_Index_Raw']
		.rolling(window=30, min_periods=10)
		.mean()
	)

	# Time-series z-score normalization.
	mean_raw = result_df['NIFTY50_Liquidity_Index_Raw'].mean()
	std_raw = result_df['NIFTY50_Liquidity_Index_Raw'].std(ddof=0)
	if pd.notna(std_raw) and std_raw > 0:
		result_df['NIFTY50_Liquidity_Index'] = (
			result_df['NIFTY50_Liquidity_Index_Raw'] - mean_raw
		) / std_raw
	else:
		result_df['NIFTY50_Liquidity_Index'] = np.nan

	if 'ILLIQ_t' in result_df.columns:
		result_df = result_df.drop(columns=['ILLIQ_t'])
	
	# Store PCA details
	weights_abs = np.abs(loadings)
	abs_sum = float(weights_abs.sum())
	if abs_sum > 0:
		contrib = (weights_abs / abs_sum) * 100
	else:
		contrib = np.full_like(weights_abs, np.nan)

	pca_details = {
		'pca_model': pca,
		'scaler': scaler,
		'proxy_columns': available_proxies,
		'explained_variance_ratio': pca.explained_variance_ratio_,
		'components': pca.components_,
		'loadings': loadings,
		'contribution_percent': contrib,
		'mean': scaler.mean_,
		'scale': scaler.scale_
	}
	
	# Save PCA weights and scaler
	weights_df = pd.DataFrame({
		'Proxy': available_proxies,
		'PCA_Loading': loadings,
		'Absolute_Loading': weights_abs,
		'Contribution_Percent': contrib,
	})
	weights_df.to_csv(output_dir / 'nifty50_pca_weights.csv', index=False)
	print(f"PCA weights saved to: {output_dir / 'nifty50_pca_weights.csv'}")
	
	# Save scaler for future use
	with open(output_dir / 'nifty50_pca_scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)
	
	print(f"Explained Variance Ratio (PC1): {pca.explained_variance_ratio_[0]:.4f}")
	print(f"PCA Components (weights):\n{weights_df}")
	
	return result_df, pca_details


def compare_indices(
	market_index_path: str | Path,
	nifty50_index_df: pd.DataFrame,
	output_dir: str | Path = '.',
	plots_dir: str | Path | None = None,
) -> dict:
	"""
	Compare Market Liquidity Index with NIFTY50 Composite Index.
	
	Returns comparison statistics and creates visualization
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	if plots_dir is None:
		plots_dir = output_dir / 'plots'
	else:
		plots_dir = Path(plots_dir)
	plots_dir.mkdir(parents=True, exist_ok=True)
	
	# Load market index
	market_df = pd.read_csv(market_index_path)
	market_df['DATE'] = pd.to_datetime(market_df['DATE'], errors='coerce')
	
	# Rename for clarity
	nifty50_df = nifty50_index_df.copy()
	nifty50_df = nifty50_df.rename(columns={'NIFTY50_Liquidity_Index': 'NIFTY50_Index'})
	
	# Merge on DATE
	comparison_df = pd.merge(
		market_df,
		nifty50_df,
		on='DATE',
		how='inner'
	)
	
	print(f"\nCommon dates for comparison: {len(comparison_df)}")
	print(f"Date range: {comparison_df['DATE'].min()} to {comparison_df['DATE'].max()}")
	
	# Compute correlation
	correlation = comparison_df[['Market_Liquidity_Index', 'NIFTY50_Index']].corr().iloc[0, 1]
	print(f"Correlation between Market Index and NIFTY50 Index: {correlation:.4f}")
	
	# Summary statistics
	stats = {
		'market_index': comparison_df['Market_Liquidity_Index'].describe(),
		'nifty50_index': comparison_df['NIFTY50_Index'].describe(),
		'correlation': correlation,
		'common_dates': len(comparison_df)
	}
	
	print("\nMarket Index Statistics:")
	print(stats['market_index'])
	print("\nNIFTY50 Index Statistics:")
	print(stats['nifty50_index'])
	
	# Essential visualization: normalized overlaid comparison.
	fig, ax = plt.subplots(figsize=(14, 6))
	
	# Normalize both indices for direct comparison
	market_norm = (comparison_df['Market_Liquidity_Index'] - comparison_df['Market_Liquidity_Index'].mean()) / comparison_df['Market_Liquidity_Index'].std()
	nifty50_norm = (comparison_df['NIFTY50_Index'] - comparison_df['NIFTY50_Index'].mean()) / comparison_df['NIFTY50_Index'].std()
	
	ax.plot(comparison_df['DATE'], market_norm, label='Market Index (Normalized)', linewidth=1.5)
	ax.plot(comparison_df['DATE'], nifty50_norm, label='NIFTY50 Index (Normalized)', linewidth=1.5)
	ax.set_title('Normalized Liquidity Index Comparison: Market vs NIFTY50', fontsize=13, fontweight='bold')
	ax.set_xlabel('Date')
	ax.set_ylabel('Normalized Index')
	ax.legend(fontsize=11)
	ax.grid(alpha=0.3)
	
	plt.tight_layout()
	overlay_path = plots_dir / 'nifty50_vs_market_overlay.png'
	plt.savefig(overlay_path, dpi=150)
	print(f"Overlay comparison saved: {overlay_path}")
	plt.close()

	# Clean up non-essential plots from prior runs to avoid confusion.
	for stale_plot in [
		plots_dir / 'nifty50_vs_market_timeseries.png',
		plots_dir / 'nifty50_vs_market_scatter.png',
		output_dir / 'nifty50_vs_market_timeseries.png',
		output_dir / 'nifty50_vs_market_scatter.png',
	]:
		if stale_plot.exists():
			stale_plot.unlink()
	
	return {'comparison_df': comparison_df, 'stats': stats}


def part2_creating_nifty50_composite_index() -> pd.DataFrame:
	"""
	Part 2: Compute NIFTY50 composite liquidity index.
	Returns nifty50_index_df for use in subsequent parts.
	"""
	print("\n" + "="*80)
	print("PART 2: CREATING NIFTY50 COMPOSITE LIQUIDITY INDEX")
	print("="*80)
	
	project_root = Path(__file__).parent.parent
	nifty50_raw_path = project_root / 'NIFTY50.csv'
	nifty50_proxies_path = project_root / 'Code' / 'nifty50_liquidity_proxies.csv'
	market_index_path = project_root / 'Code' / 'market_liquidity_index.csv'
	output_path = project_root / 'Code' / 'nifty50_liquidity_index.csv'
	output_dir = project_root / 'Code'
	plots_dir = output_dir / 'plots'
	
	print("="*70)
	print("Computing NIFTY50 Proxies + Composite Liquidity Index")
	print("="*70)
	
	# Compute NIFTY50 proxies from raw index CSV.
	print(f"\nLoading raw NIFTY50 data from: {nifty50_raw_path}")
	nifty50_ohlcv = load_and_prepare_nifty50(nifty50_raw_path)
	print(f"Loaded {len(nifty50_ohlcv)} cleaned rows")

	nifty50_proxies = compute_nifty50_liquidity_proxies(nifty50_ohlcv, drop_na_rows=True)
	nifty50_proxies.to_csv(nifty50_proxies_path, index=False)
	print(f"Saved NIFTY50 liquidity proxies: {nifty50_proxies_path}")
	
	# Create composite index via PCA
	nifty50_index_df, pca_details = create_nifty50_composite_index(
		nifty50_proxies,
		proxy_columns=['ILLIQ_t', 'DEPTH_t', 'SPREAD_t', 'REV_t'],
		n_components=1,
		output_dir=output_dir
	)
	
	# Save NIFTY50 index
	nifty50_index_df[['DATE', 'NIFTY50_Liquidity_Index']].to_csv(output_path, index=False)
	print(f"\nNIFTY50 Composite Index saved to: {output_path}")
	
	# Compare with market index
	print("\n" + "="*70)
	print("Comparing Market Index with NIFTY50 Benchmark Index")
	print("="*70)
	comparison_results = compare_indices(
		market_index_path,
		nifty50_index_df,
		output_dir=output_dir,
		plots_dir=plots_dir,
	)
	
	# Save comparison results
	comparison_results['comparison_df'].to_csv(
		output_dir / 'market_vs_nifty50_comparison.csv',
		index=False
	)
	stale_rolling_csv = output_dir / 'nifty50_liquidity_index_30d.csv'
	if stale_rolling_csv.exists():
		stale_rolling_csv.unlink()
	print(f"\nComparison data saved to: {output_dir / 'market_vs_nifty50_comparison.csv'}")
	
	print("\n" + "="*70)
	print("NIFTY50 Composite Index Creation Complete!")
	print("="*70)
	print(f"\nOutput Files:")
	print(f"  - Proxies: {nifty50_proxies_path}")
	print(f"  - Index: {output_path}")
	print(f"  - PCA Weights: {output_dir / 'nifty50_pca_weights.csv'}")
	print(f"  - Comparison: {output_dir / 'market_vs_nifty50_comparison.csv'}")
	print(f"  - Essential Visualization: {plots_dir / 'nifty50_vs_market_overlay.png'}")
	
	return nifty50_index_df


# ============================================================================
# PART 3: CREATING TOP100 LIQUIDITY INDEX
# ============================================================================

def _winsorize_columns_top100(
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
		work_df = _winsorize_columns_top100(work_df, PROXY_COLUMNS, winsor_lower_q, winsor_upper_q)

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


def part3_creating_top100_index(proxy_df: pd.DataFrame, nifty50_index_df: pd.DataFrame, market_index_df: pd.DataFrame) -> None:
	"""
	Part 3: Create top 100 most liquid stocks index and comparison visualizations.
	"""
	print("\n" + "="*80)
	print("PART 3: CREATING TOP 100 LIQUIDITY INDEX AND COMPARISONS")
	print("="*80)
	
	code_dir = Path(__file__).resolve().parent
	plots_dir = code_dir / "plots"

	selected_stocks_path = code_dir / "top100_most_liquid_stocks.csv"
	top100_index_path = code_dir / "top100_liquidity_index.csv"
	top100_weights_path = code_dir / "top100_proxy_pca_weights.csv"
	compare_csv_path = code_dir / "top100_vs_nifty50_comparison.csv"

	top100_plot_path = plots_dir / "top100_liquidity_index_plot.png"
	combined_plot_path = plots_dir / "top100_vs_nifty50_overlay.png"
	six_panel_plot_path = plots_dir / "all_indices_six_panel.png"

	print("=" * 72)
	print("Top 100 Most Liquid Stocks: Composite Liquidity Index")
	print("=" * 72)

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

	comparison_df = plot_top100_vs_nifty50(top100_index_df, nifty50_index_df, combined_plot_path)
	comparison_df.to_csv(compare_csv_path, index=False)

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


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main() -> None:
	"""
	Main orchestrator: runs all three parts sequentially.
	
	Part 1: Computing Liquidity Proxies
	Part 2: Creating NIFTY50 Composite Index
	Part 3: Creating Top100 Liquidity Index
	"""
	print("\n" + "="*80)
	print("COMBINED LIQUIDITY ANALYSIS - FULL PIPELINE")
	print("="*80)

	# Part 1: Compute proxies and market index
	proxy_df, market_index_df = part1_computing_liquidity_proxies()
	if proxy_df is None or market_index_df is None:
		print("\nPart 1 failed - input file not found. Exiting.")
		return

	# Part 2: Create NIFTY50 index
	nifty50_index_df = part2_creating_nifty50_composite_index()

	# Part 3: Create Top100 index and comparisons
	part3_creating_top100_index(proxy_df, nifty50_index_df, market_index_df)

	print("\n" + "="*80)
	print("COMBINED LIQUIDITY ANALYSIS COMPLETE!")
	print("="*80)
	print("\nAll three analyses have been executed successfully in a single run.")


if __name__ == "__main__":
	main()

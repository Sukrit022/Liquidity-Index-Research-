"""
Create NIFTY50 Composite Liquidity Index using PCA.

This script mirrors the process used to create the market liquidity index,
applying PCA to NIFTY50 individual liquidity proxies to derive a composite benchmark index.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


NIFTY_START_DATE = pd.Timestamp("2013-01-01")


def _winsorize_columns(
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
	df = df[df['DATE'] >= NIFTY_START_DATE].reset_index(drop=True)
	df = df[df['VOLUME'] > 0].reset_index(drop=True)
	if df.empty:
		raise ValueError(
			f"No NIFTY50 rows with positive volume available on/after {NIFTY_START_DATE.date()}"
		)
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
		work_df = _winsorize_columns(
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


def main():
	"""Main execution: compute proxies, create NIFTY50 PCA index, and compare."""
	project_root = Path(__file__).parent.parent
	nifty50_raw_path = project_root / 'Code' / 'NIFTY50.csv'
	if not nifty50_raw_path.exists():
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
	print(f"Loaded {len(nifty50_ohlcv)} cleaned rows from {nifty50_ohlcv['DATE'].min().date()} onward")

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
	
	return nifty50_index_df, pca_details, comparison_results


if __name__ == '__main__':
	nifty50_index_df, pca_details, comparison_results = main()

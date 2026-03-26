from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2024-12-31")
START_YEAR = 2010
END_YEAR = 2024
MIN_RATIO = 0.95


@dataclass
class StockYearMetrics:
	stock: str
	year: int
	available_days: int
	traded_days: int
	total_market_days: int
	availability_ratio: float
	trading_ratio: float


def _load_stock_file(csv_file: Path) -> pd.DataFrame:
	"""Load one stock file with only required columns and normalize types."""
	df = pd.read_csv(csv_file, usecols=["DATE", "SYMBOL", "VOLUME"])
	df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
	df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce").fillna(0)
	df = df.dropna(subset=["DATE"]).copy()

	# Keep only date window needed for index construction.
	df = df[(df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)]
	if df.empty:
		return df

	# Ensure unique day-level records if duplicates exist.
	df = df.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last")
	return df


def _load_stock_file_for_liquidity(csv_file: Path) -> pd.DataFrame:
	"""Load one stock file with required liquidity-proxy columns and normalize names."""
	df = pd.read_csv(
		csv_file,
		usecols=[
			"DATE",
			"OPEN",
			"HIGH",
			"LOW",
			"PREV. CLOSE",
			"CLOSE",
			"VWAP",
			"VOLUME",
			"VALUE",
			"NO OF TRADES",
			"SYMBOL",
		],
	)

	df = df.rename(columns={"PREV. CLOSE": "PREV_CLOSE", "NO OF TRADES": "NO_OF_TRADES"})
	df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

	numeric_cols = [
		"OPEN",
		"HIGH",
		"LOW",
		"PREV_CLOSE",
		"CLOSE",
		"VWAP",
		"VOLUME",
		"VALUE",
		"NO_OF_TRADES",
	]
	df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

	df = df.dropna(subset=["DATE", "SYMBOL"]).copy()
	df = df[(df["DATE"] >= START_DATE) & (df["DATE"] <= END_DATE)]
	if df.empty:
		return df

	# Ensure one record per stock-date.
	df = df.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last")
	return df


def _build_market_calendar(stock_data_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
	"""Build year-wise market trading days from union of available dates across stocks."""
	all_dates = []
	for df in stock_data_map.values():
		if not df.empty:
			all_dates.extend(df["DATE"].tolist())

	if not all_dates:
		return pd.DataFrame(columns=["DATE", "year"])

	market_dates = pd.DataFrame({"DATE": pd.to_datetime(pd.Series(all_dates).drop_duplicates())})
	market_dates["year"] = market_dates["DATE"].dt.year
	market_dates = market_dates[(market_dates["year"] >= START_YEAR) & (market_dates["year"] <= END_YEAR)]
	return market_dates.sort_values("DATE").reset_index(drop=True)


def _compute_metrics_for_stock(
	stock: str,
	df: pd.DataFrame,
	market_days_per_year: pd.Series,
	market_dates_by_year: dict[int, set[pd.Timestamp]],
) -> list[StockYearMetrics]:
	"""Compute yearly availability and trading ratios for one stock."""
	metrics: list[StockYearMetrics] = []

	if df.empty:
		for year in range(START_YEAR, END_YEAR + 1):
			total_days = int(market_days_per_year.get(year, 0))
			metrics.append(
				StockYearMetrics(stock, year, 0, 0, total_days, 0.0 if total_days else 0.0, 0.0 if total_days else 0.0)
			)
		return metrics

	df = df.copy()
	df["year"] = df["DATE"].dt.year

	stock_dates_by_year = {
		year: set(group["DATE"].tolist())
		for year, group in df.groupby("year")
	}
	traded_dates_by_year = {
		year: set(group.loc[group["VOLUME"] > 0, "DATE"].tolist())
		for year, group in df.groupby("year")
	}

	for year in range(START_YEAR, END_YEAR + 1):
		total_days = int(market_days_per_year.get(year, 0))
		if total_days == 0:
			metrics.append(StockYearMetrics(stock, year, 0, 0, 0, 0.0, 0.0))
			continue

		market_dates_this_year = market_dates_by_year.get(year, set())
		stock_dates_this_year = stock_dates_by_year.get(year, set())
		traded_dates_this_year = traded_dates_by_year.get(year, set())

		available_days = len(stock_dates_this_year.intersection(market_dates_this_year))
		traded_days = len(traded_dates_this_year.intersection(market_dates_this_year))

		availability_ratio = available_days / total_days
		trading_ratio = traded_days / total_days

		metrics.append(
			StockYearMetrics(
				stock=stock,
				year=year,
				available_days=available_days,
				traded_days=traded_days,
				total_market_days=total_days,
				availability_ratio=availability_ratio,
				trading_ratio=trading_ratio,
			)
		)

	return metrics


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	output_dir = Path(__file__).resolve().parent
	data_folder = project_root / "NSE_500 Data"

	if not data_folder.exists():
		raise FileNotFoundError(f"Data folder not found: {data_folder}")

	stock_files = sorted(data_folder.glob("*_historical_data.csv"))
	if not stock_files:
		raise FileNotFoundError(f"No stock files found in: {data_folder}")

	stock_data_map: dict[str, pd.DataFrame] = {}
	for csv_file in stock_files:
		stock_key = csv_file.name.replace("_historical_data.csv", "")
		try:
			stock_data_map[stock_key] = _load_stock_file(csv_file)
		except ValueError:
			# Skip malformed files with missing required columns.
			stock_data_map[stock_key] = pd.DataFrame(columns=["DATE", "SYMBOL", "VOLUME"])

	market_calendar = _build_market_calendar(stock_data_map)
	if market_calendar.empty:
		raise ValueError("No valid market dates found in the selected window (2010-2024).")

	market_days_per_year = market_calendar.groupby("year")["DATE"].nunique()
	market_dates_by_year = {
		year: set(group["DATE"].tolist())
		for year, group in market_calendar.groupby("year")
	}
	evaluation_years = sorted([int(y) for y, n in market_days_per_year.items() if n > 0])
	if not evaluation_years:
		raise ValueError("No valid evaluation years found with market-day data.")

	all_metrics: list[StockYearMetrics] = []
	for stock, df in stock_data_map.items():
		all_metrics.extend(_compute_metrics_for_stock(stock, df, market_days_per_year, market_dates_by_year))

	metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])

	# Step 1: Data availability from Jan 2010 onward with minimal gaps.
	# Rule:
	# - Stock must have at least one observation in Jan 2010.
	# - Availability ratio >= 95% in every year from 2010 to 2024.
	stock_has_jan2010 = {}
	for stock, df in stock_data_map.items():
		if df.empty:
			stock_has_jan2010[stock] = False
			continue
		jan_2010_mask = (df["DATE"] >= START_DATE) & (df["DATE"] < pd.Timestamp("2010-02-01"))
		stock_has_jan2010[stock] = bool(jan_2010_mask.any())

	availability_pass = (
		metrics_df[metrics_df["year"].isin(evaluation_years)]
		.groupby("stock")["availability_ratio"]
		.min()
		.ge(MIN_RATIO)
	)
	filter1_stocks = sorted(
		[
			stock
			for stock, passed in availability_pass.items()
			if passed and stock_has_jan2010.get(stock, False)
		]
	)

	# Step 2: Trading consistency in every year 2010-2024.
	trading_pass = (
		metrics_df[metrics_df["year"].isin(evaluation_years)]
		.groupby("stock")["trading_ratio"]
		.min()
		.ge(MIN_RATIO)
	)
	filter2_stocks = sorted([stock for stock in filter1_stocks if trading_pass.get(stock, False)])

	initial_count = len(stock_data_map)
	after_filter1_count = len(filter1_stocks)
	after_filter2_count = len(filter2_stocks)

	# Keep only the stock list used for liquidity computation.
	filtered_stocks_df = pd.DataFrame({"stock": filter2_stocks})
	filtered_stocks_path = output_dir / "filtered_stocks_2010_2024.csv"
	filtered_stocks_df.to_csv(filtered_stocks_path, index=False)

	# Build combined dataset for liquidity-proxy computation using retained stocks only.
	liquidity_frames = []
	for stock in filter2_stocks:
		csv_file = data_folder / f"{stock}_historical_data.csv"
		if not csv_file.exists():
			continue
		try:
			liquidity_df = _load_stock_file_for_liquidity(csv_file)
		except ValueError:
			continue
		if liquidity_df.empty:
			continue
		liquidity_frames.append(liquidity_df)

	if liquidity_frames:
		combined_liquidity_df = pd.concat(liquidity_frames, ignore_index=True)
		combined_liquidity_df = combined_liquidity_df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)
	else:
		combined_liquidity_df = pd.DataFrame(
			columns=[
				"DATE",
				"OPEN",
				"HIGH",
				"LOW",
				"PREV_CLOSE",
				"CLOSE",
				"VWAP",
				"VOLUME",
				"VALUE",
				"NO_OF_TRADES",
				"SYMBOL",
			]
		)

	combined_liquidity_path = output_dir / "combined_nse_daily_data.csv"
	combined_liquidity_df.to_csv(combined_liquidity_path, index=False)

	# Remove old intermediate outputs so this script leaves only two CSV outputs.
	for stale_file in [
		output_dir / "cleaned_liquidity_universe_2010_2024.csv",
		output_dir / "retained_stocks_2010_2024.csv",
		output_dir / "filter_summary_2010_2024.csv",
		output_dir / "stock_yearly_metrics_2010_2024.csv",
	]:
		if stale_file.exists():
			stale_file.unlink()

	print("Filter pipeline complete.")
	print(f"Evaluation years with available market data: {evaluation_years[0]}-{evaluation_years[-1]}")
	print(f"Initial stocks: {initial_count}")
	print(f"After Filter 1: {after_filter1_count}")
	print(f"After Filter 2: {after_filter2_count}")
	print(f"Saved filtered stock list: {filtered_stocks_path}")
	print(f"Saved combined liquidity dataset: {combined_liquidity_path}")


if __name__ == "__main__":
	main()

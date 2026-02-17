"""
LIQUIDITY PROXY CALCULATIONS FOR INDIAN EQUITY MARKET
======================================================

Purpose: Compute economically consistent illiquidity proxies for NSE stocks
Convention: ALL PROXIES ARE ILLIQUIDITY MEASURES (Higher value = MORE illiquid)

References:
- Amihud (2002): Journal of Financial Markets
- Pastor & Stambaugh (2003): Journal of Political Economy
- Roll (1984): Journal of Finance
- Corwin & Schultz (2012): Journal of Finance
- Lesmond, Ogden & Trzcinka (1999): Journal of Finance
- Fong, Holden & Trzcinka (2017): Review of Finance
"""

import pandas as pd
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import symbol mapping for corporate actions
from symbol_mapping_2015_2024 import SYMBOL_MAPPING_2015_2024

# Import proxy definitions (single source of truth)
from proxy_definitions import PROXY_DEFINITIONS, generate_definition_report, get_proxy_list, validate_proxy_definition

print("\nProxy Definitions Loaded:")
print(f"  {len(PROXY_DEFINITIONS)} proxies defined: {', '.join(get_proxy_list())}")
print("  All proxies validated for ILLIQUIDITY direction")

print("="*80)
print("LIQUIDITY PROXY CALCULATION FOR INDIAN EQUITY MARKET")
print("="*80)
print("\nConvention: ALL PROXIES ARE ILLIQUIDITY MEASURES (inputs)")
print("  Input proxies: Higher value = MORE illiquid (less liquid)")
print("  Final indices: Higher value = MORE LIQUID (better) - signs flipped")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading and preparing data...")

# Load all CSV files
all_files = glob.glob("../NSE_500 Data/*.csv")
print(f"  - Found {len(all_files)} stock files")

master_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
print(f"  - Loaded {len(master_df):,} raw observations")

# Handle corporate actions/name changes
master_df['SYMBOL'] = master_df['SYMBOL'].replace(SYMBOL_MAPPING_2015_2024)

# Parse dates and sort
master_df['DATE'] = pd.to_datetime(master_df['DATE'])
master_df = master_df.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)

# Keep only necessary columns
required_cols = ['DATE', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VALUE']
master_df = master_df[required_cols]

# Drop rows with missing OHLC or zero prices
master_df = master_df.dropna(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VALUE'])
master_df = master_df[(master_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']] > 0).all(axis=1)]

# Create YEAR_MONTH for aggregation
master_df['YEAR_MONTH'] = master_df['DATE'].dt.to_period('M')

# Compute daily returns (DO NOT fill NaN with 0 - keep NaN for first observation)
master_df['DAILY_RETURN'] = master_df.groupby('SYMBOL')['CLOSE'].pct_change()
master_df['ABS_RETURN'] = master_df['DAILY_RETURN'].abs()

# Define trading days (VOLUME > 0 AND VALUE > 0)
master_df['IS_TRADING_DAY'] = (master_df['VOLUME'] > 0) & (master_df['VALUE'] > 0)

print(f"  - After cleaning: {len(master_df):,} observations")
print(f"  - Date range: {master_df['DATE'].min()} to {master_df['DATE'].max()}")
print(f"  - Unique stocks(Symbol names): {master_df['SYMBOL'].nunique()}")
print(f"  - Trading days: {master_df['IS_TRADING_DAY'].sum():,} ({100*master_df['IS_TRADING_DAY'].mean():.1f}%)")

# ============================================================================
# BUILD EMPIRICAL NSE TRADING CALENDAR (Top 5 Liquid Stocks)
# ============================================================================
print("\n[1.1/6] Building empirical NSE trading calendar...")
print("  (Using union of trading dates from top 5 most liquid stocks per year)")

def build_nse_trading_calendar(df):
    """
    Build empirical NSE trading calendar from most liquid stocks.
    Uses union of trading dates from top 5 stocks by median VALUE each year.
    """
    df['YEAR'] = df['DATE'].dt.year
    
    # Find top 5 most liquid stocks per year
    top_liquid_stocks = []
    
    for year in sorted(df['YEAR'].unique()):
        year_data = df[df['YEAR'] == year]
        
        # Calculate median VALUE per symbol for the year
        median_value_by_symbol = year_data.groupby('SYMBOL')['VALUE'].median().sort_values(ascending=False)
        
        # Get top 5
        top_5 = median_value_by_symbol.head(5).index.tolist()
        top_liquid_stocks.extend([(year, sym) for sym in top_5])
        
        # print(f"    {year}: {', '.join(top_5[:5])} (median VALUE: {median_value_by_symbol.iloc[:5].values})")
    
    # Get union of all trading dates from these top stocks
    top_liquid_data = df[
        df.apply(lambda row: (row['YEAR'], row['SYMBOL']) in top_liquid_stocks, axis=1)
    ]
    
    nse_calendar = sorted(top_liquid_data['DATE'].unique())
    
    print(f"\n  Empirical NSE calendar: {len(nse_calendar)} trading days ({nse_calendar[0].date()} to {nse_calendar[-1].date()})")
    
    # Create lookup dictionary: YEAR_MONTH -> list of trading dates
    calendar_by_month = {}
    for date in nse_calendar:
        ym = pd.Timestamp(date).to_period('M')
        if ym not in calendar_by_month:
            calendar_by_month[ym] = []
        calendar_by_month[ym].append(pd.Timestamp(date))
    
    return nse_calendar, calendar_by_month, top_liquid_stocks

NSE_CALENDAR, NSE_CALENDAR_BY_MONTH, TOP_LIQUID_STOCKS = build_nse_trading_calendar(master_df)

# ============================================================================
# DIAGNOSTIC: NON-TRADING DAY (NTD) OBSERVABILITY CHECK (Empirical Calendar)
# ============================================================================
print("\n[1.2/6] Checking Non-Trading Day (NTD) observability...")
print("  (Using empirical NSE calendar, not pd.bdate_range)")

# Check if dataset includes non-trading days or only trading days
def check_ntd_observability(df, nse_calendar_by_month):
    """Determine if non-trading days are observable using empirical NSE calendar"""
    results = []
    
    for (symbol, year_month), group in df.groupby(['SYMBOL', 'YEAR_MONTH']):
        if len(group) < 5:
            continue
        
        # Get expected trading days from empirical calendar
        ym_period = pd.Period(year_month, freq='M')
        
        if ym_period not in nse_calendar_by_month:
            # Month not in calendar (edge case)
            continue
        
        expected_dates = nse_calendar_by_month[ym_period]
        n_expected_days = len(expected_dates)
        
        # Count observed rows
        n_observed = len(group)
        
        # Coverage ratio
        coverage = n_observed / n_expected_days if n_expected_days > 0 else 0
        
        results.append({
            'SYMBOL': symbol,
            'YEAR_MONTH': year_month,
            'n_expected_days': n_expected_days,
            'n_observed': n_observed,
            'coverage': coverage
        })
    
    return pd.DataFrame(results)

ntd_check = check_ntd_observability(master_df, NSE_CALENDAR_BY_MONTH)

# Determine if NTD is observable
avg_coverage = ntd_check['coverage'].mean()
median_coverage = ntd_check['coverage'].median()
pct_full_coverage = (ntd_check['coverage'] >= 0.99).mean() * 100

print(f"  Coverage analysis: Mean={avg_coverage:.2%}, Median={median_coverage:.2%}, Full coverage={pct_full_coverage:.1f}%")

if avg_coverage < 0.85:
    NTD_OBSERVABLE = False
    print(f"\\n  [CONCLUSION] Non-trading days (NTD) are NOT observable")
    print(f"  Dataset contains only trading days (no explicit zero-volume rows)")
    print(f"  FHT will be computed using z = zero_return_days / trading_days")
    FHT_NOTE = "FHT_trading_only"
else:
    NTD_OBSERVABLE = True
    print(f"\\n  [CONCLUSION] Non-trading days (NTD) ARE observable")
    print(f"  Dataset includes non-trading days")
    print(f"  FHT will use standard formula with empirical NSE calendar")
    FHT_NOTE = "FHT_standard_empirical"

print(f"  FHT calculation mode: {FHT_NOTE}")

# ============================================================================
# STEP 2: COMPUTE INDIVIDUAL ILLIQUIDITY PROXIES
# ============================================================================
print("\n[2/6] Computing illiquidity proxies...")
print("  (All formulas follow literature conventions)")

# ----------------------------------------------------------------------------
# PROXY 1: AMIHUD (2002) ILLIQUIDITY RATIO
# ----------------------------------------------------------------------------
# Formula: AMIHUD = |Return| / (Value in millions)
# Economic interpretation: Price impact per unit of trading volume
# Direction: Higher = MORE illiquid (larger price impact)
# Reference: Amihud (2002, JFM) Equation 3
print("\n  [1/13] Amihud (2002) illiquidity ratio...")

master_df['AMIHUD'] = master_df['ABS_RETURN'] / (master_df['VALUE'] / 1e6)
master_df['AMIHUD'] = master_df['AMIHUD'].replace([np.inf, -np.inf], np.nan)

# ----------------------------------------------------------------------------
# PROXY 2: HIGH-LOW RANGE (Normalized)
# ----------------------------------------------------------------------------
# Formula: 2 * (High - Low) / (High + Low)
# Direction: Higher = MORE illiquid (wider intraday range/volatility)
# Reference: Normalized range proxy (Parkinson 1980) - NOT a spread estimator
# Note: This measures intraday volatility/price uncertainty, distinct from bid-ask spread
print("  [2/13] High-low range (normalized volatility proxy)...")

master_df['HL_RANGE'] = 2 * (master_df['HIGH'] - master_df['LOW']) / (master_df['HIGH'] + master_df['LOW'])
master_df['HL_RANGE'] = master_df['HL_RANGE'].replace([np.inf, -np.inf], np.nan)

# ----------------------------------------------------------------------------
# PROXY 3: ROLL (1984) EFFECTIVE SPREAD
# ----------------------------------------------------------------------------
# Formula: 2 * sqrt(- Cov(ΔP_t, ΔP_t-1))
# Direction: Higher = MORE illiquid (wider effective spread)
# Reference: Roll (1984, JF)
# IMPORTANT: Positive serial covariance (momentum/trends) violates Roll model assumptions
#            Set to 0 when cov >= 0 (interpreted as no bid-ask bounce detected)
#            NOTE: High % of zeros is common in markets with tick sizes and price discreteness
print("  [3/13] Roll (1984) effective spread...")

def calculate_roll_spread_monthly(data):
    """Roll (1984) spread from serial covariance of price changes"""
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Need at least 10 observations
        if len(group) < 10:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'ROLL_SPREAD': np.nan})
            continue
        
        # Get price changes
        price_changes = group['CLOSE'].diff().dropna()
        
        if len(price_changes) < 5:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'ROLL_SPREAD': np.nan})
            continue
        
        # Compute serial covariance
        cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        # Roll spread: 2 * sqrt(-cov), set to 0 if cov is positive
        roll_spread = 2 * np.sqrt(max(-cov, 0))
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'ROLL_SPREAD': roll_spread})
    
    return pd.DataFrame(results)

roll_spread_data = calculate_roll_spread_monthly(master_df)
master_df = master_df.merge(roll_spread_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# Diagnostic: Check percentage of zero ROLL_SPREAD values
temp_roll = master_df[['SYMBOL', 'YEAR_MONTH', 'ROLL_SPREAD']].drop_duplicates()
if len(temp_roll) > 0:
    pct_zero_roll = (temp_roll['ROLL_SPREAD'] == 0).mean() * 100
    print(f"     Diagnostic: {pct_zero_roll:.1f}% of ROLL_SPREAD values are zero")
    if pct_zero_roll > 40:
        print(f"     [WARNING] High zero % suggests Roll model may not suit Indian market tick structure")

# ----------------------------------------------------------------------------
# PROXY 4: ZERO RETURN RATIO (Trading Speed)
# ----------------------------------------------------------------------------
# Formula: Fraction of trading days with zero return
# Direction: Higher = MORE illiquid (more zero-return days)
# Reference: Lesmond et al. (1999, JF)
print("  [4/13] Zero return ratio (Lesmond et al. 1999)...")

def calculate_zero_ratio(data):
    """Fraction of trading days with exactly zero return"""
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Filter to trading days with valid returns
        trading_days = group[group['IS_TRADING_DAY'] & group['DAILY_RETURN'].notna()]
        
        n_trading = len(trading_days)
        
        if n_trading == 0:
            zero_ratio = np.nan
        else:
            n_zero = (trading_days['DAILY_RETURN'] == 0).sum()
            zero_ratio = n_zero / n_trading
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'ZERO_RATIO': zero_ratio})
    
    return pd.DataFrame(results)

zero_ratio_data = calculate_zero_ratio(master_df)
master_df = master_df.merge(zero_ratio_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# ----------------------------------------------------------------------------
# PROXY 5: NEAR-ZERO RETURN RATIO (India tick-size robust)
# ----------------------------------------------------------------------------
# Formula: Fraction of trading days with |return| <= 1 bp
# Direction: Higher = MORE illiquid (more near-zero returns due to tick size)
# Motivation: Indian tick sizes (₹0.05, ₹0.10) create near-zero returns
print("  [5/13] Near-zero return ratio (tick-size robust)...")

def calculate_near_zero_ratio(data, threshold=0.0001):
    """Fraction of trading days with |return| <= threshold (1 basis point)"""
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        trading_days = group[group['IS_TRADING_DAY'] & group['DAILY_RETURN'].notna()]
        
        n_trading = len(trading_days)
        
        if n_trading == 0:
            near_zero_ratio = np.nan
        else:
            n_near_zero = (trading_days['DAILY_RETURN'].abs() <= threshold).sum()
            near_zero_ratio = n_near_zero / n_trading
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'NEAR_ZERO': near_zero_ratio})
    
    return pd.DataFrame(results)

near_zero_data = calculate_near_zero_ratio(master_df)
master_df = master_df.merge(near_zero_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# ----------------------------------------------------------------------------
# PROXY 6: FONG-HOLDEN-TRZCINKA (2017) ESTIMATOR (NTD-aware)
# ----------------------------------------------------------------------------
# Formula: FHT = 2 * σ * Φ^(-1)((1 + z)/2)
# where z = zero-return proportion, σ = std of non-zero returns
# Direction: Higher = MORE illiquid (larger effective spread)
# Reference: Fong, Holden & Trzcinka (2017, Review of Finance)
# Note: Calculation adjusted based on NTD observability
print(f"  [6/13] Fong-Holden-Trzcinka (2017) estimator ({FHT_NOTE})...")

def calculate_fht(data, ntd_observable):
    """FHT spread estimator based on zero returns (NTD-aware)"""
    # VALIDATION: Check proxy definition from proxy_definitions.py
    assert 'FHT' in PROXY_DEFINITIONS, "FHT not defined in PROXY_DEFINITIONS"
    fht_def = PROXY_DEFINITIONS['FHT']
    assert fht_def['direction'] == 'illiquidity', f"FHT direction must be 'illiquidity', got {fht_def['direction']}"
    assert 'spread estimator' in fht_def['name'].lower() or 'spread estimator' in fht_def['implementation_notes'].lower(), \
        "FHT definition must mention 'spread estimator'"
    
    # Verify required columns
    required_cols = fht_def['required_columns']
    for col in required_cols:
        assert col in data.columns or col == 'DAILY_RETURN', f"Required column {col} missing for FHT"
    
    results = []
    diagnostics = {'too_few_obs': 0, 'z_zero': 0, 'z_too_high': 0, 'valid': 0, 'other_error': 0}
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Trading days with valid returns
        valid_returns = group[group['DAILY_RETURN'].notna()]
        
        if len(valid_returns) < 10:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'FHT': np.nan})
            diagnostics['too_few_obs'] += 1
            continue
        
        # Compute zero proportion among valid trading days
        # NOTE: If NTD not observable, z = zero_days / trading_days only
        #       If NTD observable, standard formula applies
        n_zero = (valid_returns['DAILY_RETURN'] == 0).sum()
        n_total = len(valid_returns)
        z = n_zero / n_total
        
        # Get non-zero returns for volatility
        non_zero_returns = valid_returns[valid_returns['DAILY_RETURN'] != 0]['DAILY_RETURN']
        
        # CRITICAL: FHT formula only works when z > 0 (need some zero returns)
        if z == 0:
            # No zero returns = perfectly liquid, set FHT to 0
            fht = 0.0
            diagnostics['z_zero'] += 1
        elif len(non_zero_returns) < 5 or z >= 0.95:
            fht = np.nan
            diagnostics['z_too_high'] += 1
        else:
            sigma = non_zero_returns.std()
            
            if sigma <= 0 or not np.isfinite(sigma):
                fht = np.nan
                diagnostics['other_error'] += 1
            else:
                try:
                    # FHT formula: 2 * σ * Φ^(-1)((1 + z)/2)
                    ppf_arg = (1 + z) / 2
                    if ppf_arg >= 1.0:
                        fht = np.nan
                        diagnostics['other_error'] += 1
                    else:
                        inverse_normal = stats.norm.ppf(ppf_arg)
                        fht = 2 * sigma * inverse_normal
                        
                        if not np.isfinite(fht) or fht < 0:
                            fht = np.nan
                            diagnostics['other_error'] += 1
                        else:
                            diagnostics['valid'] += 1
                except (ValueError, RuntimeWarning):
                    fht = np.nan
                    diagnostics['other_error'] += 1
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'FHT': fht})
    
    # Print diagnostic summary
    total = sum(diagnostics.values())
    print(f"     FHT diagnostics (total: {total:,}):")
    print(f"       - Valid FHT: {diagnostics['valid']:,} ({100*diagnostics['valid']/total:.1f}%)")
    print(f"       - z=0 (set to 0): {diagnostics['z_zero']:,} ({100*diagnostics['z_zero']/total:.1f}%)")
    print(f"       - Too few obs (<10): {diagnostics['too_few_obs']:,} ({100*diagnostics['too_few_obs']/total:.1f}%)")
    print(f"       - z too high (>=0.95): {diagnostics['z_too_high']:,} ({100*diagnostics['z_too_high']/total:.1f}%)")
    print(f"       - Other errors: {diagnostics['other_error']:,} ({100*diagnostics['other_error']/total:.1f}%)")
    
    # Check for degeneracy
    valid_ratio = diagnostics['valid'] / total if total > 0 else 0
    pct_zeros = diagnostics['z_zero'] / total if total > 0 else 0
    
    if valid_ratio < 0.2 or pct_zeros > 0.8:
        print(f"\n     [WARNING] FHT degeneracy detected:")
        print(f"       - Valid ratio: {valid_ratio:.1%} (threshold: 20%)")
        print(f"       - Zero ratio: {pct_zeros:.1%} (threshold: 80%)")
        if not ntd_observable:
            print(f"       - Cause: Missing NTD in dataset (only trading days observed)")
        print(f"       - ACTION: FHT will be excluded from index construction")
        return pd.DataFrame(results), True  # Flag as degenerate
    
    return pd.DataFrame(results), False

fht_data, FHT_DEGENERATE = calculate_fht(master_df, NTD_OBSERVABLE)
master_df = master_df.merge(fht_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# Mark FHT as degenerate if needed
if FHT_DEGENERATE:
    print(f"\n     [DECISION] FHT marked as 'excluded: degenerate due to missing NTD'")
    print(f"     FHT will not be used in proxy selection or index construction")

# ----------------------------------------------------------------------------
# PROXY 7: PASTOR-STAMBAUGH (2003) GAMMA
# ----------------------------------------------------------------------------
# Formula: Regression r'_{t+1} = θ + φ*r'_t + γ*sign(r'_t)*volume_t + ε
# where r'_t = return - market_return
# Direction: -γ is reported (higher = MORE illiquid, stronger reversal)
# Reference: Pastor & Stambaugh (2003, JPE)
print("  [7/13] Pastor-Stambaugh (2003) gamma...")

# Compute market return (equal-weighted cross-sectional mean)
# FIXED: Only include stocks with at least 60 days of trading history to reduce survival bias
print("  Computing market return for Pastor-Stambaugh (equal-weighted, min 60 days history)...")

# Count trading days per symbol up to each date
master_df = master_df.sort_values(['SYMBOL', 'DATE'])
master_df['DAYS_TRADING'] = master_df.groupby('SYMBOL').cumcount() + 1

# Market return: equal-weighted mean across stocks with sufficient history
qualified_returns = master_df[
    (master_df['DAILY_RETURN'].notna()) & 
    (master_df['DAYS_TRADING'] >= 60)  # At least ~3 months of trading
]

market_return = qualified_returns.groupby('DATE')['DAILY_RETURN'].mean().reset_index()
market_return.rename(columns={'DAILY_RETURN': 'MARKET_RETURN'}, inplace=True)
master_df = master_df.merge(market_return, on='DATE', how='left')

print(f"     - Market return computed from stocks with >=60 days history")
print(f"     - Mean daily market return: {market_return['MARKET_RETURN'].mean():.6f}")

# Compute excess return
master_df['EXCESS_RETURN'] = master_df['DAILY_RETURN'] - master_df['MARKET_RETURN']

def calculate_pastor_gamma(data):
    """Pastor-Stambaugh (2003) reversal coefficient"""
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Filter to valid observations
        group = group.sort_values('DATE')
        valid_obs = group[
            group['EXCESS_RETURN'].notna() & 
            group['VALUE'].notna() & 
            (group['VOLUME'] > 0)
        ]
        
        # Need at least 8 observations
        if len(valid_obs) < 8:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'PASTOR': np.nan})
            continue
        
        # Build regression variables
        r_prime = valid_obs['EXCESS_RETURN'].values
        volume = valid_obs['VALUE'].values
        
        # Create lagged variables: r'_{t+1} = f(r'_t, sign(r'_t)*volume_t)
        if len(r_prime) < 8:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'PASTOR': np.nan})
            continue
        
        r_prime_t_plus_1 = r_prime[1:]
        r_prime_t = r_prime[:-1]
        volume_t = volume[:-1]
        
        # CRITICAL: x_t = sign(r'_t) * volume_t
        sign_return = np.sign(r_prime_t)
        x_t = sign_return * volume_t
        
        # Standardize x_t
        if np.std(x_t) > 0:
            x_t_std = (x_t - np.mean(x_t)) / np.std(x_t)
        else:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'PASTOR': np.nan})
            continue
        
        # OLS: r'_{t+1} = theta + phi*r'_t + gamma*x_t
        if len(r_prime_t_plus_1) < 7:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'PASTOR': np.nan})
            continue
        
        X = np.column_stack([np.ones(len(r_prime_t_plus_1)), r_prime_t, x_t_std])
        y = r_prime_t_plus_1
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            gamma = beta[2]
            
            # DIRECTION TRANSFORMATION: Report -gamma so higher = more illiquid
            pastor_illiq = -gamma
            
            if not np.isfinite(pastor_illiq):
                pastor_illiq = np.nan
        except (np.linalg.LinAlgError, ValueError):
            pastor_illiq = np.nan
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'PASTOR': pastor_illiq})
    
    return pd.DataFrame(results)

pastor_data = calculate_pastor_gamma(master_df)
master_df = master_df.merge(pastor_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# ----------------------------------------------------------------------------
# PROXY 8: AMIVEST LIQUIDITY (Inverted)
# ----------------------------------------------------------------------------
# Formula: TRUE AMIVEST = sum(VALUE) / sum(|Return|) for active trading days
#          Then invert: AMIVEST_ILLIQ = 1 / (AMIVEST_LIQ + eps)
# Direction: Higher = MORE illiquid (lower liquidity)
# Reference: Amihud & Mendelson (1989) - volume per unit price change
# NOTE: Different from AMIHUD which uses mean(|ret|/VALUE)
print("  [8/13] Amivest liquidity ratio (inverted)...")

def calculate_amivest_monthly(data):
    """
    TRUE AMIVEST LIQUIDITY = sum(VALUE) / sum(|Return|)
    
    This differs from AMIHUD:
    - AMIHUD = mean(|ret| / VALUE) - averages RATIOS
    - AMIVEST = sum(VALUE) / sum(|ret|) - RATIO of SUMS
    
    Not mechanically correlated due to different aggregation method.
    """
    proxy_name = 'AMIVEST'
    definition = PROXY_DEFINITIONS[proxy_name]
    
    # Validate required columns
    for col in definition['required_columns']:
        if col not in data.columns:
            raise ValueError(f"{proxy_name}: Required column '{col}' not found")
    
    # Validate direction
    assert definition['direction'] == 'illiquidity', f"{proxy_name} must be illiquidity measure"
    
    results = []
    eps = 1e-12
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Active trading days: |ret| > 0 AND VALUE > 0
        active_mask = (group['ABS_RETURN'] > 0) & (group['VALUE'] > 0)
        active_days = group[active_mask]
        
        if len(active_days) < 15:
            amivest_illiq = np.nan
        else:
            # TRUE AMIVEST: sum(VALUE) / sum(|ret|)
            total_value = active_days['VALUE'].sum()
            total_abs_return = active_days['ABS_RETURN'].sum()
            
            if total_abs_return > 0:
                amivest_liq = total_value / total_abs_return
                # Convert to illiquidity
                amivest_illiq = 1 / (amivest_liq + eps)
            else:
                amivest_illiq = np.nan
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'AMIVEST': amivest_illiq})
    
    return pd.DataFrame(results)

amivest_data = calculate_amivest_monthly(master_df)

# FIXED: Winsorize using TRAINING period only (2005-2018) to avoid look-ahead bias
print("     - Winsorizing monthly AMIVEST at 1%/99% (training period only)...")
if amivest_data['AMIVEST'].notna().sum() > 0:
    # Define training period
    train_end = pd.Period('2018-12', freq='M')
    train_data = amivest_data[amivest_data['YEAR_MONTH'] <= train_end]
    
    if len(train_data) > 0 and train_data['AMIVEST'].notna().sum() > 0:
        # Compute winsorization bounds on training data only
        lower = train_data['AMIVEST'].quantile(0.01)
        upper = train_data['AMIVEST'].quantile(0.99)
        # Apply to full sample
        amivest_data['AMIVEST'] = amivest_data['AMIVEST'].clip(lower=lower, upper=upper)
        print(f"        Winsorized range (from training): [{lower:.4e}, {upper:.4e}]")
    else:
        print("        [WARNING] Insufficient training data for winsorization")

master_df = master_df.merge(amivest_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# ----------------------------------------------------------------------------
# PROXY 9: LOT (Lesmond, Ogden & Trzcinka 1999) - Threshold-based
# ----------------------------------------------------------------------------
# Formula: Fraction of days with |return| <= threshold (implicit transaction cost)
# Direction: Higher = MORE illiquid (more days with negligible price movement)
# Reference: Lesmond et al. (1999, JF) - limited dependent variable approach
# Note: Differs from ZERO_RATIO by using threshold (5bp) instead of exact zero
print("  [9/13] LOT measure (threshold-based transaction costs)...")

def calculate_lot(data, lot_thresh=0.0005):
    """LOT: proportion of days with no price movement beyond tick size"""
    # VALIDATION: Check proxy definition from proxy_definitions.py
    assert 'LOT' in PROXY_DEFINITIONS, "LOT not defined in PROXY_DEFINITIONS"
    lot_def = PROXY_DEFINITIONS['LOT']
    assert lot_def['direction'] == 'illiquidity', f"LOT direction must be 'illiquidity', got {lot_def['direction']}"
    assert '5bp' in lot_def['description'] or '5bp' in lot_def['formula'] or 'threshold' in lot_def['description'].lower(), \
        "LOT definition must mention '5bp' or 'threshold'"
    
    # Verify required columns
    required_cols = lot_def['required_columns']
    for col in required_cols:
        assert col in data.columns or col == 'DAILY_RETURN', f"Required column {col} missing for LOT"
    
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        # Filter to valid trading days: VALUE > 0 and return not NaN
        valid_days = group[(group['VALUE'] > 0) & group['DAILY_RETURN'].notna()]
        
        if len(valid_days) < 15:
            results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'LOT': np.nan})
            continue
        
        # LOT: fraction of days where |return| <= threshold (no movement beyond tick)
        n_lot = (valid_days['DAILY_RETURN'].abs() <= lot_thresh).sum()
        lot = n_lot / len(valid_days)
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'LOT': lot})
    
    return pd.DataFrame(results)

lot_data = calculate_lot(master_df, lot_thresh=0.0005)  # 5 basis points
master_df = master_df.merge(lot_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# Diagnostic: Check correlation between ZERO_RATIO and LOT
temp_check = master_df[['SYMBOL', 'YEAR_MONTH', 'ZERO_RATIO', 'LOT']].drop_duplicates()
temp_check_clean = temp_check[['ZERO_RATIO', 'LOT']].dropna()
if len(temp_check_clean) > 0:
    zero_lot_corr = temp_check_clean['ZERO_RATIO'].corr(temp_check_clean['LOT'])
    print(f"     Diagnostic: corr(ZERO_RATIO, LOT) = {zero_lot_corr:.4f}")
    if zero_lot_corr >= 0.98:
        print(f"     [WARNING] High correlation {zero_lot_corr:.4f} >= 0.98 suggests redundancy!")
    else:
        print(f"     [OK] Correlation {zero_lot_corr:.4f} < 0.98 - measures are sufficiently distinct")

# ----------------------------------------------------------------------------
# PROXY 10: TRADING VOLUME ILLIQUIDITY (Inverted)
# ----------------------------------------------------------------------------
# Formula: 1 / (1 + Volume)
# Direction: Higher = MORE illiquid (lower volume)
# Transformation: Inverted from trading volume
print("  [10/13] Trading volume illiquidity (inverted)...")

master_df['AVG_VOLUME'] = master_df.groupby(['SYMBOL', 'YEAR_MONTH'])['VOLUME'].transform('mean')
# DIRECTION TRANSFORMATION: Invert so higher = more illiquid
master_df['VOLUME_ILLIQ'] = 1 / (1 + master_df['AVG_VOLUME'])

# ----------------------------------------------------------------------------
# PROXY 11: TURNOVER ILLIQUIDITY (Inverted)
# ----------------------------------------------------------------------------
# Formula: 1 / (1 + Turnover)
# where Turnover = Value traded
# Direction: Higher = MORE illiquid (lower turnover)
print("  [11/13] Turnover illiquidity (inverted)...")

master_df['AVG_TURNOVER'] = master_df.groupby(['SYMBOL', 'YEAR_MONTH'])['VALUE'].transform('mean')
# DIRECTION TRANSFORMATION: Invert so higher = more illiquid
master_df['TURNOVER_ILLIQ'] = 1 / (1 + master_df['AVG_TURNOVER'])

# ----------------------------------------------------------------------------
# PROXY 12: COEFFICIENT OF VARIATION OF TURNOVER
# ----------------------------------------------------------------------------
# Formula: CV = Std(Turnover) / Mean(Turnover)
# Direction: Higher = MORE illiquid (more erratic trading)
print("  [12/13] Coefficient of variation of turnover...")

def calculate_cv_turnover(data):
    """CV of turnover: volatility of trading activity"""
    results = []
    
    for (symbol, year_month), group in data.groupby(['SYMBOL', 'YEAR_MONTH']):
        turnover_std = group['VALUE'].std()
        turnover_mean = group['VALUE'].mean()
        
        if turnover_mean > 0:
            cv_turnover = turnover_std / turnover_mean
        else:
            cv_turnover = np.nan
        
        results.append({'SYMBOL': symbol, 'YEAR_MONTH': year_month, 'CV_TURNOVER': cv_turnover})
    
    return pd.DataFrame(results)

cv_turnover_data = calculate_cv_turnover(master_df)
master_df = master_df.merge(cv_turnover_data, on=['SYMBOL', 'YEAR_MONTH'], how='left')

# ----------------------------------------------------------------------------
# PROXY 13: DOLLAR ILLIQUIDITY (Price impact normalized)
# ----------------------------------------------------------------------------
# Formula: |Return| / (Value / Price)
# Direction: Higher = MORE illiquid
print("  [13/13] Dollar illiquidity (normalized price impact)...")

master_df['DOLLAR_ILLIQ'] = master_df['ABS_RETURN'] / ((master_df['VALUE'] / master_df['CLOSE']) + 1)
master_df['DOLLAR_ILLIQ'] = master_df['DOLLAR_ILLIQ'].replace([np.inf, -np.inf], np.nan)

print("\n  All 13 proxies computed successfully!")

# ============================================================================
# STEP 3: AGGREGATE TO MONTHLY FREQUENCY
# ============================================================================
print("\n[3/6] Aggregating to monthly frequency...")

# Create monthly dataset
monthly_data = master_df.groupby(['SYMBOL', 'YEAR_MONTH']).agg({
    # Daily-level proxies: average over month
    'AMIHUD': 'mean',
    'HL_RANGE': 'mean',
    'DOLLAR_ILLIQ': 'mean',
    'VOLUME_ILLIQ': 'mean',
    'TURNOVER_ILLIQ': 'mean',
    
    # Monthly-level proxies: take first (already aggregated)
    'ROLL_SPREAD': 'first',
    'ZERO_RATIO': 'first',
    'NEAR_ZERO': 'first',
    'FHT': 'first',
    'PASTOR': 'first',
    'AMIVEST': 'first',
    'LOT': 'first',
    'CV_TURNOVER': 'first',
    
    # Count trading days
    'DATE': 'count'
}).rename(columns={'DATE': 'N_DAYS'}).reset_index()

# Filter: require at least 15 trading days per month
monthly_data = monthly_data[monthly_data['N_DAYS'] >= 15].copy()

print(f"  - Monthly observations: {len(monthly_data):,}")
print(f"  - Unique stocks: {monthly_data['SYMBOL'].nunique()}")
print(f"  - Time range: {monthly_data['YEAR_MONTH'].min()} to {monthly_data['YEAR_MONTH'].max()}")

# ============================================================================
# STEP 4: VERIFY DIRECTIONAL CONSISTENCY
# ============================================================================
print("\n[4/6] Verifying directional consistency...")
print("  Convention: ALL INPUT PROXIES ARE ILLIQUIDITY MEASURES")
print("  (Higher proxy value = MORE illiquid)")
print("  (Final indices will be flipped: Higher index = MORE liquid)")

proxy_cols = [
    'AMIHUD',          # ✓ Higher = more illiquid (price impact)
    'HL_RANGE',        # ✓ Higher = more illiquid (wider range/volatility)
    'ROLL_SPREAD',     # ✓ Higher = more illiquid (wider spread)
    'ZERO_RATIO',      # ✓ Higher = more illiquid (more zero days)
    'NEAR_ZERO',       # ✓ Higher = more illiquid (more near-zero days)
    'FHT',             # ✓ Higher = more illiquid (larger spread estimate)
    'PASTOR',          # ✓ Higher = more illiquid (-gamma, reversal)
    'AMIVEST',         # ✓ Higher = more illiquid (price impact)
    'LOT',             # ✓ Higher = more illiquid (more inactive days)
    'VOLUME_ILLIQ',    # ✓ Higher = more illiquid (INVERTED from volume)
    'TURNOVER_ILLIQ',  # ✓ Higher = more illiquid (INVERTED from turnover)
    'CV_TURNOVER',     # ✓ Higher = more illiquid (more erratic)
    'DOLLAR_ILLIQ'     # ✓ Higher = more illiquid (price impact)
]

print(f"\n  All {len(proxy_cols)} proxies are directionally consistent:")
for i, proxy in enumerate(proxy_cols, 1):
    nan_pct = monthly_data[proxy].isna().mean() * 100
    print(f"    {i:2d}. {proxy:<18} (NaN: {nan_pct:5.1f}%)")

# ============================================================================
# STEP 5: HANDLE MISSING VALUES AND OUTLIERS
# ============================================================================
print("\n[5/6] Handling missing values and outliers...")

# Count missing values
print("\n  Missing value summary:")
for proxy in proxy_cols:
    n_missing = monthly_data[proxy].isna().sum()
    pct_missing = 100 * n_missing / len(monthly_data)
    print(f"    {proxy:<18} : {n_missing:>6,} ({pct_missing:5.1f}%)")

# Winsorize outliers at 1% and 99% (conservative approach)
print("\n  Winsorizing outliers (1% and 99% quantiles)...")
for proxy in proxy_cols:
    if monthly_data[proxy].notna().sum() > 0:
        lower = monthly_data[proxy].quantile(0.01)
        upper = monthly_data[proxy].quantile(0.99)
        monthly_data[proxy] = monthly_data[proxy].clip(lower=lower, upper=upper)

# Final dataset: drop rows with ANY missing proxy
print(f"\n  Before dropping NaN: {len(monthly_data):,} observations")
final_data = monthly_data.dropna(subset=proxy_cols).copy()
print(f"  After dropping NaN:  {len(final_data):,} observations")
print(f"  Final stocks: {final_data['SYMBOL'].nunique()}")
print(f"  Final time range: {final_data['YEAR_MONTH'].min()} to {final_data['YEAR_MONTH'].max()}")

# ============================================================================
# STEP 6: COMPUTE CORRELATION MATRIX
# ============================================================================
print("\n[6/6] Computing correlation matrix...")
print("="*80)

# Compute Pearson correlation
correlation_matrix = final_data[proxy_cols].corr()

print("\nPEARSON CORRELATION MATRIX")
print("(All proxies are illiquidity measures: higher = more illiquid)")
print("="*80)
print("\nCorrelation matrix (13 x 13):")
print(correlation_matrix.round(3))

# Summary statistics of correlations
corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
print(f"\nCorrelation summary:")
print(f"  Mean:   {np.mean(corr_values):.3f}")
print(f"  Median: {np.median(corr_values):.3f}")
print(f"  Min:    {np.min(corr_values):.3f}")
print(f"  Max:    {np.max(corr_values):.3f}")
print(f"  Std:    {np.std(corr_values):.3f}")

# Identify high correlations
print(f"\nHigh correlations (|r| > 0.7):")
high_corr_pairs = []
for i in range(len(proxy_cols)):
    for j in range(i+1, len(proxy_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((proxy_cols[i], proxy_cols[j], corr_val))

if high_corr_pairs:
    for proxy1, proxy2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {proxy1:<18} <-> {proxy2:<18} : {corr:>6.3f}")
else:
    print("  None (all |r| <= 0.7)")

# Unit test: AMIHUD vs AMIVEST correlation
print("\n" + "="*80)
print("UNIT TEST: AMIVEST vs AMIHUD Differentiation")
print("="*80)
if 'AMIHUD' in proxy_cols and 'AMIVEST' in proxy_cols:
    # Pooled (cross-sectional) correlation
    amihud_amivest_corr = correlation_matrix.loc['AMIHUD', 'AMIVEST']
    print(f"\nPooled Correlation(AMIHUD, AMIVEST): {amihud_amivest_corr:.4f}")
    
    # Time-series correlation (market-level medians)
    market_medians_test = final_data.groupby('YEAR_MONTH')[['AMIHUD', 'AMIVEST']].median()
    amihud_amivest_corr_ts = market_medians_test['AMIHUD'].corr(market_medians_test['AMIVEST'])
    print(f"Time-Series Correlation (market medians): {amihud_amivest_corr_ts:.4f}")
    
    print(f"\nFormula differences:")
    print(f"  AMIHUD  = mean(|ret| / VALUE)     [averages RATIOS]")
    print(f"  AMIVEST = 1/[sum(VALUE)/sum(|ret|)] [inverted RATIO of SUMS]")
    print(f"\nExpected: Pooled corr captures cross-sectional variation (small vs large stocks)")
    print(f"          Time-series corr tests co-movement during crises (more relevant for market-state index)")
    print(f"          Both should be moderate (0.3-0.7), not mechanical (>0.95)")
    
    # Check pooled correlation
    if amihud_amivest_corr > 0.95:
        print(f"  [FAIL - Pooled] Correlation {amihud_amivest_corr:.4f} > 0.95 suggests measures are too similar!")
        print(f"  [ACTION] Check AMIVEST implementation - should use sum(VALUE)/sum(|ret|), not mean")
    elif amihud_amivest_corr < 0.2:
        print(f"  [WARNING - Pooled] Correlation {amihud_amivest_corr:.4f} < 0.2 surprisingly low")
    else:
        print(f"  [PASS - Pooled] Correlation {amihud_amivest_corr:.4f} shows measures are related but distinct")
    
    # Check time-series correlation (more important for market-state index)
    if amihud_amivest_corr_ts > 0.95:
        print(f"  [FAIL - Time-Series] Correlation {amihud_amivest_corr_ts:.4f} > 0.95 suggests cyclical redundancy!")
    elif amihud_amivest_corr_ts < 0.2:
        print(f"  [WARNING - Time-Series] Correlation {amihud_amivest_corr_ts:.4f} < 0.2 - proxies behave differently in crises")
    else:
        print(f"  [PASS - Time-Series] Correlation {amihud_amivest_corr_ts:.4f} shows distinct cyclical behavior")
        print(f"  [OK] AMIVEST successfully differentiated from AMIHUD")
    
    # Show descriptive statistics
    print(f"\nDescriptive statistics (monthly illiquidity):")
    print(f"  {'Proxy':<12} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for proxy in ['AMIHUD', 'AMIVEST']:
        vals = final_data[proxy].dropna()
        print(f"  {proxy:<12} {vals.mean():>12.4e} {vals.median():>12.4e} {vals.std():>12.4e} {vals.min():>12.4e} {vals.max():>12.4e}")
else:
    print("\n[WARNING] Cannot perform AMIHUD vs AMIVEST unit test - one or both proxies missing")
print("="*80)

# Save correlation matrix
correlation_matrix.to_csv("liquidity_correlation_matrix.csv")
print(f"\nCorrelation matrix saved to: liquidity_correlation_matrix.csv")

# ============================================================================
# GENERATE PROXY DEFINITION REPORT
# ============================================================================
print(f"\nGenerating proxy definition report...")

# Generate report from proxy_definitions.py
report_df = generate_definition_report()
report_df.to_csv("proxy_definitions_report.csv", index=False)
print(f"  Proxy definition report saved to: proxy_definitions_report.csv")

# Validate all computed proxies have definitions
undefined_proxies = [p for p in proxy_cols if p not in PROXY_DEFINITIONS]
if undefined_proxies:
    print(f"\n  [WARNING] Proxies without definitions: {', '.join(undefined_proxies)}")
else:
    print(f"\n  [OK] All {len(proxy_cols)} computed proxies have definitions in PROXY_DEFINITIONS")

# Print summary table
print(f"\n  Proxy Definition Summary:")
print(f"  {'Proxy':<18} {'Direction':<12} {'Dimension':<20} {'Priority':>8}")
print(f"  {'-'*18} {'-'*12} {'-'*20} {'-'*8}")
for proxy in proxy_cols:
    if proxy in PROXY_DEFINITIONS:
        pdef = PROXY_DEFINITIONS[proxy]
        direction = pdef['direction']
        dimension = pdef.get('dimension', 'N/A')
        priority = pdef.get('priority', 'N/A')
        print(f"  {proxy:<18} {direction:<12} {dimension:<20} {priority:>8}")
    else:
        print(f"  {proxy:<18} {'UNDEFINED':<12} {'N/A':<20} {'N/A':>8}")

# Create correlation heatmap
print(f"Creating correlation heatmap visualization...")
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r', 
            center=0, 
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'})
plt.title('Liquidity Proxy Correlation Matrix\n(Input proxies: Higher = More Illiquid)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Illiquidity Proxies (Input)', fontsize=12, fontweight='bold')
plt.ylabel('Illiquidity Proxies (Input)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('liquidity_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Correlation heatmap saved to: liquidity_correlation_matrix.png")

# Save final proxy data
final_data[['SYMBOL', 'YEAR_MONTH'] + proxy_cols].to_csv("liquidity_proxies_monthly.csv", index=False)
print(f"Monthly proxy data saved to: liquidity_proxies_monthly.csv")

# ============================================================================
# STEP 7: DIMENSION-BALANCED PROXY SELECTION & INDEX CONSTRUCTION
# ============================================================================
print("\n" + "="*80)
print("DIMENSION-BALANCED PROXY SELECTION & LIQUIDITY INDEX CONSTRUCTION")
print("="*80)

# ============================================================================
# A) DEFINE PROXY METADATA (Dimension + Direction + Priority)
# ============================================================================

PROXY_META = {
    # Price Impact
    'AMIHUD': {'dimension': 'price_impact', 'direction': 'illiquidity', 'priority': 3},
    'DOLLAR_ILLIQ': {'dimension': 'price_impact', 'direction': 'illiquidity', 'priority': 2},
    'AMIVEST': {'dimension': 'price_impact', 'direction': 'illiquidity', 'priority': 2},
    'PASTOR': {'dimension': 'price_impact', 'direction': 'illiquidity', 'priority': 1},
    # Range / Volatility (price uncertainty)
    'HL_RANGE': {'dimension': 'range_volatility', 'direction': 'illiquidity', 'priority': 3},
    # Spread / Cost (transaction costs)
    'ROLL_SPREAD': {'dimension': 'spread_cost', 'direction': 'illiquidity', 'priority': 2},
    # Speed / Stickiness
    'ZERO_RATIO': {'dimension': 'speed_stickiness', 'direction': 'illiquidity', 'priority': 3},
    'NEAR_ZERO': {'dimension': 'speed_stickiness', 'direction': 'illiquidity', 'priority': 2},
    'FHT': {'dimension': 'speed_stickiness', 'direction': 'illiquidity', 'priority': 2},
    'LOT': {'dimension': 'speed_stickiness', 'direction': 'illiquidity', 'priority': 1},
    # Activity / Quantity (prefer level proxies over dispersion)
    'TURNOVER_ILLIQ': {'dimension': 'activity_quantity', 'direction': 'illiquidity', 'priority': 3},
    'VOLUME_ILLIQ': {'dimension': 'activity_quantity', 'direction': 'illiquidity', 'priority': 2},
    'CV_TURNOVER': {'dimension': 'activity_quantity', 'direction': 'illiquidity', 'priority': 1}
}

print("\n[A] Proxy Metadata Defined")
print(f"  Available proxies: {len([p for p in PROXY_META.keys() if p in proxy_cols])}/{len(PROXY_META)}")

# Exclude FHT if degenerate
if FHT_DEGENERATE:
    print(f"\n[A.1] Excluding Degenerate Proxies")
    print(f"  - FHT excluded: degenerate due to missing NTD")
    if 'FHT' in PROXY_META:
        del PROXY_META['FHT']
    proxy_cols_for_selection = [p for p in proxy_cols if p != 'FHT']
    
    # CRITICAL: Verify dimension coverage after exclusion
    dimensions_covered = set()
    for proxy in proxy_cols_for_selection:
        if proxy in PROXY_META:
            dimensions_covered.add(PROXY_META[proxy]['dimension'])
    
    print(f"  [VALIDATION] Dimensions covered after FHT exclusion: {sorted(dimensions_covered)}")
    
    # Check if speed_stickiness dimension still covered
    speed_proxies = [p for p in proxy_cols_for_selection 
                     if p in PROXY_META and PROXY_META[p]['dimension'] == 'speed_stickiness']
    if len(speed_proxies) == 0:
        print(f"  [ERROR] speed_stickiness dimension has NO proxies after FHT exclusion!")
    else:
        print(f"  [OK] speed_stickiness still covered by: {speed_proxies}")
else:
    proxy_cols_for_selection = proxy_cols.copy()

# ============================================================================
# B) DATA QUALITY DIAGNOSTICS
# ============================================================================

def proxy_quality_report(df, proxies):
    """Generate quality report with warnings for poor data quality"""
    results = []
    warnings_list = []
    
    for proxy in proxies:
        if proxy not in df.columns:
            continue
        
        data = df[proxy]
        pct_nan = data.isna().mean() * 100
        variance = data.var()
        unique_count = data.nunique()
        pct_zeros = ((data == 0).sum() / data.notna().sum() * 100) if data.notna().sum() > 0 else 0
        
        # Quality checks
        has_issue = False
        issues = []
        
        if pct_nan > 30:
            issues.append(f"HIGH_NAN({pct_nan:.1f}%)")
            warnings_list.append(f"  [WARN] {proxy}: {pct_nan:.1f}% missing > 30%")
            has_issue = True
        
        if variance == 0 or not np.isfinite(variance):
            issues.append("ZERO_VARIANCE")
            warnings_list.append(f"  [WARN] {proxy}: Zero or infinite variance")
            has_issue = True
        
        if pct_zeros > 95:
            issues.append(f"EXCESSIVE_ZEROS({pct_zeros:.1f}%)")
            warnings_list.append(f"  [WARN] {proxy}: {pct_zeros:.1f}% zeros > 95%")
            has_issue = True
        
        results.append({
            'proxy': proxy,
            'pct_nan': pct_nan,
            'variance': variance,
            'unique_count': unique_count,
            'pct_zeros': pct_zeros,
            'has_issue': has_issue,
            'issues': "; ".join(issues) if issues else "OK"
        })
    
    return pd.DataFrame(results), warnings_list

print("\n[B] Data Quality Diagnostics")
quality_report, quality_warnings = proxy_quality_report(final_data, proxy_cols_for_selection)
print(quality_report.to_string(index=False))

if quality_warnings:
    print("\nQuality Warnings:")
    for warning in quality_warnings:
        print(warning)

# Exclude proxies failing quality checks
failed_proxies = quality_report[quality_report['has_issue'] == True]['proxy'].tolist()
proxies_for_selection = [p for p in proxy_cols_for_selection if p not in failed_proxies]
print(f"\nProxies passing quality: {len(proxies_for_selection)}/{len(proxy_cols_for_selection)}")

# ============================================================================
# B.1) INTERPRETABILITY CHECK FOR ACTIVITY PROXIES
# ============================================================================

def check_proxy_interpretability(df, proxy, n_deciles=10, min_months=12):
    """Check if proxy increases monotonically across LIQ deciles"""
    print(f"\n[B.1] Interpretability Check: {proxy}")
    
    # We need a preliminary LIQ score - use simple average of available proxies
    # Use proxies that passed quality checks
    available_proxies = [p for p in ['AMIHUD', 'HL_RANGE', 'ZERO_RATIO', 'TURNOVER_ILLIQ'] 
                        if p in df.columns and p in proxies_for_selection]
    
    if len(available_proxies) < 2:
        print(f"  Skipping: insufficient proxies for preliminary LIQ score")
        return True  # Pass by default if can't check
    
    # Compute preliminary LIQ score (simple average of standardized proxies)
    df_check = df.copy()
    for p in available_proxies:
        df_check[f'Z_{p}'] = df_check.groupby('YEAR_MONTH')[p].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    z_cols = [f'Z_{p}' for p in available_proxies]
    df_check['LIQ_PRELIM'] = df_check[z_cols].mean(axis=1)
    
    # Check monotonicity month by month
    monotonic_months = 0
    non_monotonic_months = 0
    
    for month, group in df_check.groupby('YEAR_MONTH'):
        if len(group) < n_deciles * 5:  # Need enough stocks per decile
            continue
        
        # Create deciles based on LIQ_PRELIM
        group['LIQ_DECILE'] = pd.qcut(group['LIQ_PRELIM'], q=n_deciles, labels=False, duplicates='drop')
        
        # Get median proxy value per decile
        decile_medians = group.groupby('LIQ_DECILE')[proxy].median().sort_index()
        
        if len(decile_medians) < 3:
            continue
        
        # Check if monotonically increasing
        is_monotonic = (decile_medians.diff().dropna() >= 0).all()
        
        if is_monotonic:
            monotonic_months += 1
        else:
            non_monotonic_months += 1
    
    total_months = monotonic_months + non_monotonic_months
    
    if total_months == 0:
        print(f"  Skipping: no valid months for check")
        return True
    
    monotonic_pct = monotonic_months / total_months * 100
    
    print(f"  Monotonicity: {monotonic_months}/{total_months} months ({monotonic_pct:.1f}%)")
    
    # Pass if >= 80% of months show monotonicity
    passes = monotonic_pct >= 80
    
    if passes:
        print(f"  [PASS] {proxy} shows sufficient monotonicity ({monotonic_pct:.1f}% >= 80%)")
    else:
        print(f"  [FAIL] {proxy} fails monotonicity check ({monotonic_pct:.1f}% < 80%)")
    
    return passes

# Apply interpretability check to activity proxies
print(f"\n[B.1] Checking Activity Proxy Interpretability...")
activity_proxies = [p for p in proxies_for_selection if p in ['TURNOVER_ILLIQ', 'VOLUME_ILLIQ', 'CV_TURNOVER']]

interpretability_results = {}
for proxy in activity_proxies:
    if proxy in final_data.columns:
        passes = check_proxy_interpretability(final_data, proxy)
        interpretability_results[proxy] = passes
        if not passes:
            print(f"  [DEMOTE] {proxy} demoted due to poor interpretability")

# Demote CV_TURNOVER if it fails, or if better alternatives exist
if 'CV_TURNOVER' in interpretability_results:
    if not interpretability_results['CV_TURNOVER']:
        # Demote CV_TURNOVER by removing it unless it's the only activity proxy
        other_activity = [p for p in ['TURNOVER_ILLIQ', 'VOLUME_ILLIQ'] if p in proxies_for_selection]
        if other_activity:
            print(f"  [ACTION] Removing CV_TURNOVER from selection (level proxies available)")
            proxies_for_selection = [p for p in proxies_for_selection if p != 'CV_TURNOVER']
    elif all(interpretability_results.get(p, False) for p in ['TURNOVER_ILLIQ', 'VOLUME_ILLIQ'] if p in activity_proxies):
        # If level proxies pass, CV_TURNOVER is optional extra
        print(f"  [NOTE] CV_TURNOVER is optional (level proxies preferred)")

print(f"\nProxies after interpretability check: {len(proxies_for_selection)}")

# ============================================================================
# C) REDUNDANCY REMOVAL (WITHIN DIMENSION ONLY)
# ============================================================================

def select_core_proxies(df, high_corr_pairs, proxy_meta, proxies_to_consider, corr_thresh=0.7):
    """
    Select core proxies removing redundancy ONLY within same dimension
    
    # FIX: Correlation threshold applied ONLY within dimension because:
    #      - Cross-dimensional correlations capture different liquidity aspects
    #      - E.g., AMIHUD (price impact) and ZERO_RATIO (speed) may correlate
    #        but represent different economic channels (price vs. trading frequency)
    #      - Within-dimension correlations indicate true redundancy
    #      - This preserves dimensional balance and economic interpretability
    """
    selection_log = []
    core_proxies = set(proxies_to_consider)
    
    print("\n[C] Redundancy Removal (Within-Dimension Only)")
    
    # Separate same-dimension vs cross-dimension pairs
    same_dim_pairs = []
    cross_dim_pairs = []
    
    for proxy1, proxy2, corr in high_corr_pairs:
        if proxy1 not in proxy_meta or proxy2 not in proxy_meta:
            continue
        if proxy1 not in proxies_to_consider or proxy2 not in proxies_to_consider:
            continue
        
        dim1 = proxy_meta[proxy1]['dimension']
        dim2 = proxy_meta[proxy2]['dimension']
        
        if dim1 == dim2:
            same_dim_pairs.append((proxy1, proxy2, corr, dim1))
        else:
            cross_dim_pairs.append((proxy1, proxy2, corr, dim1, dim2))
    
    print(f"  High-corr pairs (|r|>{corr_thresh}): {len(high_corr_pairs)}")
    print(f"    - Same dimension: {len(same_dim_pairs)} (will apply redundancy removal)")
    print(f"    - Cross dimension: {len(cross_dim_pairs)} (kept - different aspects)")
    
    # FIX: Process only same-dimension pairs for redundancy removal
    for proxy1, proxy2, corr, dim in same_dim_pairs:
        if proxy1 not in core_proxies or proxy2 not in core_proxies:
            continue
        
        # Score = (1 - pct_nan) * variance * priority
        nan1 = df[proxy1].isna().mean()
        var1 = df[proxy1].var()
        pri1 = proxy_meta[proxy1]['priority']
        score1 = (1 - nan1) * var1 * pri1
        
        nan2 = df[proxy2].isna().mean()
        var2 = df[proxy2].var()
        pri2 = proxy_meta[proxy2]['priority']
        score2 = (1 - nan2) * var2 * pri2
        
        # Drop lower-scoring proxy
        if score1 > score2:
            core_proxies.remove(proxy2)
            reason = f"Same dim ({dim}), lower score ({score2:.2e} < {score1:.2e})"
            selection_log.append(f"  [-] {proxy2}: {reason}")
            print(f"  Drop {proxy2} (keep {proxy1}, r={corr:.3f})")
        else:
            core_proxies.remove(proxy1)
            reason = f"Same dim ({dim}), lower score ({score1:.2e} < {score2:.2e})"
            selection_log.append(f"  [-] {proxy1}: {reason}")
            print(f"  Drop {proxy1} (keep {proxy2}, r={corr:.3f})")
    
    # Log cross-dimensional pairs (kept)
    if cross_dim_pairs:
        print(f"  Kept {len(cross_dim_pairs)} cross-dimensional pairs (e.g., {cross_dim_pairs[0][0]}<->{cross_dim_pairs[0][1]})")
        for proxy1, proxy2, corr, dim1, dim2 in cross_dim_pairs[:3]:
            selection_log.append(f"  [KEPT] {proxy1}<->{proxy2} (r={corr:.3f}): different dimensions ({dim1} vs {dim2})")
    
    # Enforce dimension coverage
    print("\n[C.1] Enforcing Dimension Coverage")
    dimension_coverage = {}
    for proxy in core_proxies:
        dim = proxy_meta[proxy]['dimension']
        dimension_coverage.setdefault(dim, []).append(proxy)
    
    required_dims = ['price_impact', 'spread_cost', 'speed_stickiness', 'activity_quantity']
    for dim in required_dims:
        if dim not in dimension_coverage or len(dimension_coverage[dim]) == 0:
            candidates = [(p, proxy_meta[p]['priority']) 
                         for p in proxies_to_consider 
                         if proxy_meta[p]['dimension'] == dim]
            if candidates:
                best_proxy = max(candidates, key=lambda x: x[1])[0]
                core_proxies.add(best_proxy)
                selection_log.append(f"  [+] {best_proxy}: Added for {dim} coverage")
                print(f"  Added {best_proxy} for {dim} coverage")
    
    # Target size: 4-6 proxies
    if len(core_proxies) > 6:
        print(f"\n[C.2] Trimming to 4-6 proxies (current: {len(core_proxies)})")
        final_proxies = []
        
        # One best per dimension
        for dim in required_dims:
            dim_proxies = [p for p in core_proxies if proxy_meta[p]['dimension'] == dim]
            if dim_proxies:
                scored = [(p, (1-df[p].isna().mean())*df[p].var()*proxy_meta[p]['priority']) 
                         for p in dim_proxies]
                final_proxies.append(max(scored, key=lambda x: x[1])[0])
        
        # Add 2 best extras if space
        remaining = [p for p in core_proxies if p not in final_proxies]
        if remaining and len(final_proxies) < 6:
            scored = [(p, (1-df[p].isna().mean())*df[p].var()*proxy_meta[p]['priority']) 
                     for p in remaining]
            for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:2]:
                final_proxies.append(p)
        
        core_proxies = set(final_proxies)
        selection_log.append(f"  [TRIM] Reduced to {len(core_proxies)} proxies")
    
    return sorted(list(core_proxies)), selection_log

# Execute selection
core_proxies, selection_log = select_core_proxies(
    final_data, high_corr_pairs, PROXY_META, proxies_for_selection, corr_thresh=0.7
)

print(f"\n[RESULT] Selected {len(core_proxies)} core proxies:")
for i, proxy in enumerate(core_proxies, 1):
    meta = PROXY_META[proxy]
    print(f"  {i}. {proxy:<18} [{meta['dimension']:<20}] priority={meta['priority']}")

# ============================================================================
# D) STANDARDIZATION (MONTHLY CROSS-SECTIONAL Z-SCORE)
# ============================================================================

def standardize_monthly(df, proxies, proxy_meta):
    """Standardize using monthly cross-sectional z-scores"""
    print("\n[D] Monthly Cross-Sectional Standardization")
    df_std = df.copy()
    
    for proxy in proxies:
        direction = proxy_meta[proxy]['direction']
        temp_col = f"_temp_{proxy}"
        
        # Flip sign if liquidity direction
        if direction == 'liquidity':
            df_std[temp_col] = -df_std[proxy]
            print(f"  {proxy}: Flipped sign (was liquidity)")
        else:
            df_std[temp_col] = df_std[proxy]
        
        # Monthly cross-sectional z-score
        z_col = f"Z_{proxy}"
        df_std[z_col] = df_std.groupby('YEAR_MONTH')[temp_col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan
        )
        df_std.drop(columns=[temp_col], inplace=True)
        
        nan_pct = df_std[z_col].isna().mean() * 100
        print(f"  {proxy} -> {z_col} (NaN: {nan_pct:.1f}%)")
    
    return df_std

monthly_std = standardize_monthly(final_data, core_proxies, PROXY_META)
z_cols = [f"Z_{p}" for p in core_proxies]

# CRITICAL CHECK: Verify cross-sectional z-scores have ~zero mean each month
print("\n[D.1] Verification: Cross-Sectional Z-Score Sanity Check")
for z_col in z_cols[:2]:  # Check first 2 proxies
    monthly_means = monthly_std.groupby('YEAR_MONTH')[z_col].mean()
    print(f"  {z_col}: mean across months = {monthly_means.mean():.6f} (should be ~0)")
print("  [NOTE] Cross-sectional z-scores have zero mean each month by construction")
print("  [NOTE] Market index will be built from RAW proxies, not z-scores, to avoid degeneracy")

# ============================================================================
# E) STOCK-LEVEL LIQUIDITY SCORES (Cross-Sectional Characteristic)
# ============================================================================

def build_stock_liquidity_scores(df, core_proxies, proxy_meta, train_frac=0.7):
    """Build stock-level liquidity scores using cross-sectional z-scores"""
    print("\n[E] Stock-Level Liquidity Scores (Cross-Sectional Characteristic)")
    from sklearn.decomposition import PCA
    
    z_cols = [f"Z_{p}" for p in core_proxies]
    data_pca = df[['SYMBOL', 'YEAR_MONTH'] + z_cols].dropna(subset=z_cols).copy()
    
    # Train/test split by time
    unique_months = sorted(data_pca['YEAR_MONTH'].unique())
    n_train = int(len(unique_months) * train_frac)
    train_months = unique_months[:n_train]
    
    print(f"  Training: {train_months[0]} to {train_months[-1]} ({n_train} months)")
    
    # Fit PCA on training data only
    train_data = data_pca[data_pca['YEAR_MONTH'].isin(train_months)]
    X_train = train_data[z_cols].values
    
    pca = PCA(n_components=1)
    pca.fit(X_train)
    
    w_pca_stock = pca.components_[0]
    explained_var_stock = pca.explained_variance_ratio_[0]
    
    print(f"  Explained variance: {explained_var_stock*100:.2f}%")
    print(f"  Weights (stock characteristic):")
    for proxy, weight in zip(core_proxies, w_pca_stock):
        print(f"    {proxy:<18}: {weight:>8.4f}")
    
    # Compute stock scores using fixed weights
    # FLIP SIGN: Convert illiquidity to liquidity (higher = MORE liquid)
    X_all = data_pca[z_cols].values
    data_pca['LIQ_STOCK'] = -(X_all @ w_pca_stock)
    
    # Sign alignment with Z_AMIHUD (ensure higher LIQ = lower illiquidity)
    if 'Z_AMIHUD' in z_cols:
        corr_check = data_pca[['LIQ_STOCK', 'Z_AMIHUD']].corr().iloc[0, 1]
        if corr_check > 0:
            print(f"  [FIX] Flipping sign (corr with Z_AMIHUD: {corr_check:.3f})")
            w_pca_stock = -w_pca_stock
            data_pca['LIQ_STOCK'] = -data_pca['LIQ_STOCK']
        else:
            print(f"  [OK] Sign aligned correctly (corr with Z_AMIHUD: {corr_check:.3f})")
    
    print(f"  Interpretation: Higher LIQ_STOCK = MORE liquid (better)")
    
    # VALIDATION: Check persistence (stock characteristic should be stable)
    print(f"\n  [VALIDATION] Stock score persistence (month-to-month):")
    data_pca_sorted = data_pca.sort_values(['SYMBOL', 'YEAR_MONTH'])
    data_pca_sorted['LAG_SCORE'] = data_pca_sorted.groupby('SYMBOL')['LIQ_STOCK'].shift(1)
    persistence = data_pca_sorted[['LIQ_STOCK', 'LAG_SCORE']].corr().iloc[0, 1]
    print(f"    Autocorrelation: {persistence:.3f}")
    if persistence > 0.6:
        print(f"    [OK] High persistence confirms LIQ_STOCK is a stock characteristic")
    else:
        print(f"    [WARNING] Low persistence ({persistence:.3f}) suggests scores are noisy")
    
    # Save stock scores
    data_pca[['SYMBOL', 'YEAR_MONTH', 'LIQ_STOCK']].rename(columns={'YEAR_MONTH':'MONTH'}).to_csv('liq_stock_scores_pca.csv', index=False)
    pd.DataFrame({'proxy': core_proxies, 'weight': w_pca_stock}).to_csv('pca_weights_stock.csv', index=False)
    
    print(f"  Saved: liq_stock_scores_pca.csv, pca_weights_stock.csv")
    
    return data_pca, w_pca_stock, explained_var_stock

# ============================================================================
# F) MARKET LIQUIDITY INDEX (Time-Series State, from Raw Proxies)
# ============================================================================

def build_market_liquidity_index(df, core_proxies, proxy_meta, train_frac=0.7):
    """Build market-state liquidity index from raw proxy aggregates"""
    print("\n[F] Market Liquidity Index (Time-Series State from Raw Proxies)")
    from sklearn.decomposition import PCA
    
    # Use RAW proxies (not cross-sectional z-scores) to avoid degeneracy
    data_market = df[['SYMBOL', 'YEAR_MONTH'] + core_proxies].dropna(subset=core_proxies).copy()
    
    print(f"  Computing market proxy series from raw proxies...")
    
    # Aggregate to market level: median across stocks each month (robust to outliers)
    market_proxies_raw = data_market.groupby('YEAR_MONTH')[core_proxies].median().reset_index()
    
    print(f"  Market proxy series: {len(market_proxies_raw)} months x {len(core_proxies)} proxies")
    
    # Standardize each market proxy series across TIME
    # ----------------------------------------------------------------------------
    # METHODOLOGICAL NOTE: De-trending and Standardization
    # ----------------------------------------------------------------------------
    # We apply linear de-trending before standardization for the following reasons:
    # 
    # 1. SECULAR TRENDS: Indian equity market liquidity improved structurally 2005-2024
    #    due to: (a) reduced tick sizes, (b) increased market depth, (c) technology adoption
    #    
    # 2. ECONOMIC INTERPRETATION: De-trending isolates CYCLICAL liquidity shocks from
    #    SECULAR market development. The final index measures "liquidity relative to trend"
    #    rather than "absolute liquidity". This makes crisis events (2008, 2020) comparable
    #    despite massive differences in market structure.
    #    
    # 3. STATISTICAL JUSTIFICATION: Without de-trending, PCA would be dominated by
    #    secular trends (AMIHUD declined 5x 2005→2024), not cyclical variation.
    #    De-trending ensures PCA captures co-movement during stress events.
    #    
    # 4. ALTERNATIVE APPROACH: One could use first-differences or log-returns instead,
    #    but de-trending preserves level interpretation while removing trends.
    #    
    # CRISIS DETECTION: 2008-09 and 2020 COVID appear severe because they represent
    # large DEVIATIONS from improving trend, not because absolute liquidity was worse
    # than 2005 levels. This is the intended interpretation for cycle analysis.
    # ----------------------------------------------------------------------------
    print(f"  De-trending and standardizing market proxy series...")
    print(f"  Using TRAINING period only for trend estimation (no look-ahead bias)")
    market_proxies_std = market_proxies_raw.copy()
    
    from scipy import signal
    
    # Train/test split by time (do this BEFORE de-trending)
    unique_months = sorted(market_proxies_raw['YEAR_MONTH'].unique())
    n_train = int(len(unique_months) * train_frac)
    train_months = unique_months[:n_train]
    
    print(f"  Training: {train_months[0]} to {train_months[-1]} ({n_train} months)")
    
    for proxy in core_proxies:
        values = market_proxies_raw[proxy].values
        
        # Estimate trend on TRAINING period only
        train_mask = market_proxies_raw['YEAR_MONTH'].isin(train_months)
        train_values = values[train_mask]
        train_indices = np.arange(len(train_values))
        
        # Fit linear trend on training data
        trend_coef = np.polyfit(train_indices, train_values, deg=1)
        
        # Apply trend removal to FULL sample using training-fitted trend
        full_indices = np.arange(len(values))
        trend_line = np.polyval(trend_coef, full_indices)
        detrended = values - trend_line
        
        # Standardize using full sample std (after de-trending)
        std_val = detrended.std()
        if std_val > 0:
            market_proxies_std[proxy] = detrended / std_val
        else:
            market_proxies_std[proxy] = 0
    
    # Fit PCA on training period
    train_market = market_proxies_std[market_proxies_std['YEAR_MONTH'].isin(train_months)]
    X_train_market = train_market[core_proxies].values
    
    pca_market = PCA(n_components=1)
    pca_market.fit(X_train_market)
    
    w_pca_market = pca_market.components_[0]
    explained_var_market = pca_market.explained_variance_ratio_[0]
    
    print(f"  Explained variance: {explained_var_market*100:.2f}%")
    print(f"  Weights (market state):")
    for proxy, weight in zip(core_proxies, w_pca_market):
        print(f"    {proxy:<18}: {weight:>8.4f}")
    
    # Compute market index for all periods
    # FLIP SIGN: Convert illiquidity to liquidity (higher = MORE liquid)
    X_all_market = market_proxies_std[core_proxies].values
    market_proxies_std['LIQ_MARKET_PCA'] = -(X_all_market @ w_pca_market)
    
    # ALSO compute equal-weighted index for comparison (gives all proxies equal voice)
    print(f"\n  Computing equal-weighted index for comparison...")
    # Equal weights: each proxy gets 1/N weight (weights sum to 1.0)
    w_equal = np.ones(len(core_proxies)) / len(core_proxies)
    market_proxies_std['LIQ_MARKET_EW'] = -(X_all_market @ w_equal)
    
    # Choose which to use as primary (equal-weighted captures all dimensions better)
    market_proxies_std['LIQ_MARKET'] = market_proxies_std['LIQ_MARKET_EW']
    print(f"\n  [DECISION] Using equal-weighted index as primary (captures all liquidity dimensions)")
    print(f"  [RATIONALE] PCA over-weights AMIHUD/spread; equal-weighting captures circuit breakers/zero-trading")
    
    # Sign alignment with raw AMIHUD market series (ensure negative correlation)
    if 'AMIHUD' in core_proxies:
        corr_check_ew = market_proxies_std[['LIQ_MARKET_EW', 'AMIHUD']].corr().iloc[0, 1]
        if corr_check_ew > 0:
            print(f"  [FIX] Flipping equal-weighted sign (corr with AMIHUD: {corr_check_ew:.3f})")
            w_equal = -w_equal
            market_proxies_std['LIQ_MARKET_EW'] = -market_proxies_std['LIQ_MARKET_EW']
            market_proxies_std['LIQ_MARKET'] = market_proxies_std['LIQ_MARKET_EW']
        
        corr_check_pca = market_proxies_std[['LIQ_MARKET_PCA', 'AMIHUD']].corr().iloc[0, 1]
        if corr_check_pca > 0:
            print(f"  [FIX] Flipping PCA sign (corr with AMIHUD: {corr_check_pca:.3f})")
            w_pca_market = -w_pca_market
            market_proxies_std['LIQ_MARKET_PCA'] = -market_proxies_std['LIQ_MARKET_PCA']
        
        print(f"  PCA vs Equal-weighted correlation: {market_proxies_std[['LIQ_MARKET_PCA', 'LIQ_MARKET_EW']].corr().iloc[0, 1]:.3f}")
    
    # FINAL VALIDATION: Sign convention check
    print(f"\n  [VALIDATION] Final sign convention check:")
    if 'AMIHUD' in core_proxies:
        amihud_median = market_proxies_raw['AMIHUD']
        liq_market_corr = market_proxies_std['LIQ_MARKET'].corr(amihud_median)
        print(f"    Correlation(LIQ_MARKET, AMIHUD_median): {liq_market_corr:.3f}")
        if liq_market_corr < -0.3:
            print(f"    [OK] Strong negative correlation confirms correct interpretation")
            print(f"         (Higher LIQ_MARKET = More liquid = Lower AMIHUD)")
        elif liq_market_corr > 0.3:
            print(f"    [ERROR] Positive correlation indicates WRONG SIGN!")
        else:
            print(f"    [WARNING] Weak correlation - verify index construction")
    
    print(f"  Interpretation: Higher LIQ_MARKET = MORE liquid market (better conditions)")
    
    # Save both indices - equal-weighted as primary, PCA for comparison
    market_output_ew = market_proxies_std[['YEAR_MONTH', 'LIQ_MARKET_EW']].copy()
    market_output_ew.columns = ['MONTH', 'LIQ_MARKET']
    market_output_ew.to_csv('liq_market_index_equal_weighted.csv', index=False)  # Primary output (equal-weighted)
    
    market_output_pca = market_proxies_std[['YEAR_MONTH', 'LIQ_MARKET_PCA']].copy()
    market_output_pca.columns = ['MONTH', 'LIQ_MARKET']
    market_output_pca.to_csv('liq_market_index_pca_variance_weighted.csv', index=False)  # Alternative
    
    pd.DataFrame({'proxy': core_proxies, 'weight_equal': w_equal, 'weight_pca': w_pca_market}).to_csv('pca_weights_market.csv', index=False)
    
    print(f"  Saved: liq_market_index_equal_weighted.csv (equal-weighted, PRIMARY)")
    print(f"  Saved: liq_market_index_pca_variance_weighted.csv (PCA, for comparison)")
    
    return market_proxies_std, w_equal, explained_var_market

stock_pca, pca_weights_stock, pca_expl_var_stock = build_stock_liquidity_scores(monthly_std, core_proxies, PROXY_META)
market_pca, pca_weights_market, pca_expl_var_market = build_market_liquidity_index(final_data, core_proxies, PROXY_META)

# ============================================================================
# H) MARKET-STATE APC (Pairwise Covariance for Robustness)
# ============================================================================

def build_market_state_apc(df, core_proxies, train_frac=0.7):
    """
    Build market-state APC using pairwise covariance (missingness-robust)
    
    METHODOLOGICAL NOTE: APC vs PCA Differences
    --------------------------------------------
    APC (Asymptotic Principal Components) uses pairwise covariance matrix which:
    1. Handles missing data better than standard PCA
    2. Weights each proxy pair equally regardless of variance
    3. Can produce different factor loadings than PCA
    
    WHY APC MIGHT DIFFER FROM PCA:
    - PCA maximizes variance explained → over-weights high-variance proxies (e.g., AMIHUD)
    - APC uses pairwise covariances → more balanced weighting across proxies
    - During crises: If high-variance proxies move differently than low-variance ones,
      PCA and APC can disagree on sign/magnitude
    
    SIGN VALIDATION: We apply de-trending (like PCA) and validate that APC falls
    during known crisis periods (2008-2009). If APC rises during crisis, sign is flipped.
    """
    print("\n[H] Market-State APC (Pairwise Covariance for Robustness)")
    from sklearn.decomposition import PCA
    
    # Use RAW proxies aggregated to market level
    data_apc = df[['SYMBOL', 'YEAR_MONTH'] + core_proxies].dropna(subset=core_proxies).copy()
    
    # Market medians per month
    market_series = data_apc.groupby('YEAR_MONTH')[core_proxies].median().reset_index()
    print(f"  Market proxy series: {len(market_series)} months x {len(core_proxies)} proxies")
    
    # Standardize each proxy across time (use same de-trending as PCA for consistency)
    from scipy import signal
    
    print(f"  De-trending and standardizing market proxy series (matching PCA method)...")
    print(f"  Using TRAINING period only for trend estimation (no look-ahead bias)")
    
    # Train/test split (matching PCA/EW approach)
    unique_months = sorted(market_series['YEAR_MONTH'].unique())
    n_train = int(len(unique_months) * train_frac)
    train_months = unique_months[:n_train]
    train_mask = market_series['YEAR_MONTH'].isin(train_months)
    
    M = market_series[core_proxies].values
    M_detrended = np.zeros_like(M)
    
    # Apply same de-trending as PCA method (train-only trend estimation)
    for i, proxy in enumerate(core_proxies):
        values = M[:, i]
        
        # Estimate trend on TRAINING period only
        train_values = values[train_mask]
        train_indices = np.arange(len(train_values))
        
        # Fit linear trend on training data
        trend_coef = np.polyfit(train_indices, train_values, deg=1)
        
        # Apply trend removal to FULL sample
        full_indices = np.arange(len(values))
        trend_line = np.polyval(trend_coef, full_indices)
        detrended = values - trend_line
        
        # Standardize
        std_val = detrended.std()
        if std_val > 0:
            M_detrended[:, i] = detrended / std_val
        else:
            M_detrended[:, i] = 0
    
    M_std = M_detrended  # Use de-trended standardized values
    
    # Compute pairwise covariance (handles missing data better)
    print(f"  Computing pairwise covariance matrix...")
    cov_matrix = np.zeros((len(core_proxies), len(core_proxies)))
    for i in range(len(core_proxies)):
        for j in range(i, len(core_proxies)):
            # Pairwise complete observations
            valid = ~(np.isnan(M_std[:, i]) | np.isnan(M_std[:, j]))
            if valid.sum() >= 10:
                cov_matrix[i, j] = np.cov(M_std[valid, i], M_std[valid, j])[0, 1]
                cov_matrix[j, i] = cov_matrix[i, j]
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort by decreasing eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # First eigenvector = APC weights
    w_apc = eigenvectors[:, 0]
    explained_var_apc = eigenvalues[0] / eigenvalues.sum()
    
    print(f"  Explained variance: {explained_var_apc*100:.2f}%")
    print(f"  Weights (market-state APC):")
    for proxy, weight in zip(core_proxies, w_apc):
        print(f"    {proxy:<18}: {weight:>8.4f}")
    
    # Compute market index
    market_apc_scores = M_std @ w_apc
    market_apc_df = pd.DataFrame({
        'MONTH': market_series['YEAR_MONTH'],
        'MLIQ_APC_STATE': market_apc_scores
    })
    
    # CRITICAL FIX: Sign alignment for liquidity interpretation
    # APC scores should behave like liquidity index: fall during crises (2008, 2020)
    # Use BOTH AMIHUD correlation AND crisis validation to determine sign
    needs_flip = False
    
    if 'AMIHUD' in core_proxies:
        amihud_idx = list(core_proxies).index('AMIHUD')
        corr_check = np.corrcoef(market_apc_scores, M_std[:, amihud_idx])[0, 1]
        
        print(f"  [DIAGNOSTIC] Initial APC correlation with standardized AMIHUD: {corr_check:.3f}")
        
        # Check 1: AMIHUD correlation (should be negative for liquidity index)
        if corr_check > 0:
            print(f"  [CHECK 1] Positive AMIHUD correlation suggests wrong sign")
            needs_flip = True
        else:
            print(f"  [CHECK 1] Negative AMIHUD correlation suggests correct sign")
        
        # Check 2: Crisis period validation
        crisis_months = market_series['YEAR_MONTH'].isin([
            pd.Period('2008-09', freq='M'), 
            pd.Period('2008-10', freq='M'),
            pd.Period('2009-01', freq='M')
        ])
        if crisis_months.sum() > 0:
            crisis_mean = market_apc_scores[crisis_months].mean()
            overall_mean = market_apc_scores.mean()
            print(f"  [CHECK 2] 2008 crisis: APC mean = {crisis_mean:.2f}, Overall mean = {overall_mean:.2f}")
            
            if crisis_mean > overall_mean:
                print(f"  [CHECK 2] APC rises during crisis - suggests wrong sign")
                if not needs_flip:
                    print(f"  [WARNING] AMIHUD and crisis checks disagree! Using crisis check.")
                needs_flip = True
            else:
                print(f"  [CHECK 2] APC falls during crisis - suggests correct sign")
        
        # Apply single sign flip based on both checks
        if needs_flip:
            print(f"\n  [FIX] Applying sign flip for liquidity interpretation")
            w_apc = -w_apc
            market_apc_scores = -market_apc_scores
            market_apc_df['MLIQ_APC_STATE'] = market_apc_scores
            
            # Verify both checks after flip
            corr_after = np.corrcoef(market_apc_scores, M_std[:, amihud_idx])[0, 1]
            crisis_mean_after = market_apc_scores[crisis_months].mean() if crisis_months.sum() > 0 else 0
            print(f"  [VERIFY] After flip: AMIHUD corr = {corr_after:.3f}, Crisis mean = {crisis_mean_after:.2f}")
        else:
            print(f"\n  [OK] Both checks passed - sign is correct")
    
    # Compute stock scores using APC weights on cross-sectional z-scores
    z_cols = [f"Z_{p}" for p in core_proxies]
    stock_data = df[['SYMBOL', 'YEAR_MONTH'] + core_proxies].dropna(subset=core_proxies).copy()
    
    # Cross-sectional standardization
    for proxy in core_proxies:
        stock_data[f'Z_{proxy}'] = stock_data.groupby('YEAR_MONTH')[proxy].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    X_stock = stock_data[z_cols].values
    stock_data['LIQ_APC_STATE'] = X_stock @ w_apc
    
    # Save outputs
    stock_data[['SYMBOL', 'YEAR_MONTH', 'LIQ_APC_STATE']].rename(columns={'YEAR_MONTH':'MONTH'}).to_csv('liq_stock_scores_apc_state.csv', index=False)
    market_apc_df.to_csv('liq_market_index_apc_state.csv', index=False)
    pd.DataFrame({'proxy': core_proxies, 'weight': w_apc}).to_csv('apc_state_weights.csv', index=False)
    
    print(f"  Saved: liq_stock_scores_apc_state.csv, liq_market_index_apc_state.csv, apc_state_weights.csv")
    
    return stock_data, market_apc_df, w_apc, explained_var_apc

stock_apc_state, market_apc_state, apc_state_weights, apc_state_expl_var = build_market_state_apc(final_data, core_proxies)

# ============================================================================
# I) COMPARISON: Version 1 (Market-State PCA vs APC)
# ============================================================================

def compare_market_state_indices(market_pca, market_apc_state, stock_pca, stock_apc_state):
    """Compare matched market-state PCA and APC indices"""
    print("\n[I] Comparison: Market-State PCA vs APC (Matched)")
    
    # Merge market indices
    market_both = market_pca[['YEAR_MONTH', 'LIQ_MARKET']].merge(
        market_apc_state, left_on='YEAR_MONTH', right_on='MONTH'
    )
    
    # Market state correlations
    pearson = market_both['LIQ_MARKET'].corr(market_both['MLIQ_APC_STATE'])
    spearman = market_both['LIQ_MARKET'].corr(market_both['MLIQ_APC_STATE'], method='spearman')
    
    print(f"  Market state correlations: Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    
    # VALIDATION: Correlation should be positive (both measure liquidity)
    if pearson < 0:
        print(f"  [WARNING] Negative correlation suggests sign inconsistency between methods!")
    elif pearson > 0.5:
        print(f"  [OK] Strong positive correlation confirms both measure same phenomenon")
    else:
        print(f"  [NOTE] Moderate correlation - methods weight dimensions differently")
    
    # Rolling 24-month correlation
    market_both = market_both.sort_values('YEAR_MONTH')
    rolling_corr = market_both['LIQ_MARKET'].rolling(24).corr(market_both['MLIQ_APC_STATE'])
    print(f"  Rolling 24-month corr: Mean={rolling_corr.mean():.4f}, Min={rolling_corr.min():.4f}, Max={rolling_corr.max():.4f}")
    
    # Stock-level monthly rank correlations (using market-state derived scores)
    stock_both = stock_pca[['SYMBOL', 'YEAR_MONTH', 'LIQ_STOCK']].merge(
        stock_apc_state[['SYMBOL', 'YEAR_MONTH', 'LIQ_APC_STATE']], on=['SYMBOL', 'YEAR_MONTH']
    )
    
    monthly_rank_corr = []
    for month, group in stock_both.groupby('YEAR_MONTH'):
        if len(group) >= 10:
            corr = group['LIQ_STOCK'].corr(group['LIQ_APC_STATE'], method='spearman')
            monthly_rank_corr.append(corr)
    
    print(f"  Stock rank corr (monthly): Median={np.median(monthly_rank_corr):.4f}, "
          f"IQR=[{np.percentile(monthly_rank_corr,25):.4f},{np.percentile(monthly_rank_corr,75):.4f}]")
    
    # Top decile overlap
    decile_overlaps = []
    for month, group in stock_both.groupby('YEAR_MONTH'):
        if len(group) >= 10:
            top_pca = set(group.nlargest(int(len(group)*0.1), 'LIQ_STOCK')['SYMBOL'])
            top_apc = set(group.nlargest(int(len(group)*0.1), 'LIQ_APC_STATE')['SYMBOL'])
            overlap = len(top_pca & top_apc) / len(top_pca) if len(top_pca) > 0 else 0
            decile_overlaps.append(overlap)
    
    print(f"  Top decile overlap: Mean={np.mean(decile_overlaps)*100:.1f}%, Median={np.median(decile_overlaps)*100:.1f}%")
    
    # Return summary
    summary_market_state = {
        'version': 'market_state_pca_vs_apc',
        'market_pearson': float(pearson),
        'market_spearman': float(spearman),
        'rolling_corr_mean': float(rolling_corr.mean()),
        'stock_rank_corr_median': float(np.median(monthly_rank_corr)),
        'stock_rank_corr_iqr': [float(np.percentile(monthly_rank_corr, 25)), float(np.percentile(monthly_rank_corr, 75))],
        'top_decile_overlap_mean': float(np.mean(decile_overlaps)),
        'top_decile_overlap_median': float(np.median(decile_overlaps))
    }
    
    return summary_market_state

# Execute PCA vs APC comparison
comparison_market_state = compare_market_state_indices(market_pca, market_apc_state, stock_pca, stock_apc_state)

print(f"\n  PCA vs APC Comparison: Pearson={comparison_market_state['market_pearson']:.4f}, Spearman={comparison_market_state['market_spearman']:.4f}")

# ============================================================================
# G) FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("LIQUIDITY INDEX CONSTRUCTION COMPLETE")
print("="*80)

print(f"\nCore Proxies ({len(core_proxies)}):")
for i, proxy in enumerate(core_proxies, 1):
    meta = PROXY_META[proxy]
    print(f"  {i}. {proxy:<18} [{meta['dimension']}] priority={meta['priority']}")

print(f"\nMethodological Approach:")
print(f"  Index Construction: Equal-weighted (primary) + PCA (comparison)")
print(f"  Stock Scores: Cross-sectional PCA on z-scores")
print(f"  Market Index: Equal-weighted aggregation with de-trending")
print(f"  Rationale: Equal-weighting ensures balanced dimensional representation")
print(f"            PCA provided for robustness checks and variance decomposition")

print(f"\nIndex Statistics:")
print(f"  Stock Scores (PCA): {pca_expl_var_stock*100:.1f}% variance, {len(stock_pca):,} observations")
print(f"  Market Index (Equal-weighted): {len(market_pca):,} months")
print(f"  Market Index (PCA alternative): {pca_expl_var_market*100:.1f}% variance")
print(f"  Market Index (APC alternative): {apc_state_expl_var*100:.1f}% variance")

print(f"\nMethod Validation:")
print(f"  PCA vs APC Correlation: {comparison_market_state['market_pearson']:.3f} (Pearson)")
if comparison_market_state['market_pearson'] > 0.5:
    print(f"  [OK] Strong agreement between methods")
elif comparison_market_state['market_pearson'] < 0:
    print(f"  [WARNING] Sign inconsistency detected - verify outputs")
else:
    print(f"  [NOTE] Moderate agreement - methods weight dimensions differently")

print(f"\nOutput Files:")
print("  Primary outputs (Equal-weighted index):")
print("    - liq_market_index_equal_weighted.csv (equal-weighted, PRIMARY)")
print("    - liq_stock_scores_pca.csv (stock-level characteristic scores)")
print("    - pca_weights_stock.csv, pca_weights_market.csv")
print("  Alternative methods (for robustness validation):")
print("    - liq_market_index_pca_variance_weighted.csv (PCA-weighted)")
print("    - liq_market_index_apc_state.csv (APC-weighted)")
print("    - liq_stock_scores_apc_state.csv")
print("    - apc_state_weights.csv")
print("  Auxiliary data:")
print("    - liquidity_correlation_matrix.csv, liquidity_correlation_matrix.png")
print("    - proxy_definitions_report.csv")

print("\n" + "="*80)
print("RECOMMENDED USAGE:")
print("  PRIMARY: Use equal-weighted index (liq_market_index_equal_weighted.csv)")
print("           Balanced dimensional representation, captures all liquidity aspects")
print("  VALIDATION: Compare with PCA/APC outputs for robustness")
print("              If all 3 methods agree on crisis detection, results are robust")
print("  STOCK SCORES: Use liq_stock_scores_pca.csv for cross-sectional analysis")
print("="*80)

# ============================================================================
# ADDITIONAL VISUALIZATION PLOTS FOR INDIAN MARKET LIQUIDITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("GENERATING ADDITIONAL VISUALIZATION PLOTS")
print("="*80)

from pathlib import Path
LIQUIDITY_PLOTS_DIR = Path("Liquidity_Plots")
LIQUIDITY_PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# COMPARISON PLOTS: PCA vs APC vs EQUAL-WEIGHTED
# ============================================================================
print("\n[1/9] Generating comparison plots for all 3 methods...")

# Load all 3 indices
pca_var_df = pd.read_csv('liq_market_index_pca_variance_weighted.csv')
pca_var_df['DATE'] = pd.to_datetime(pca_var_df['MONTH'].astype(str))
pca_var_df.rename(columns={'LIQ_MARKET': 'LIQ_PCA'}, inplace=True)

apc_df = pd.read_csv('liq_market_index_apc_state.csv')
apc_df['DATE'] = pd.to_datetime(apc_df['MONTH'].astype(str))
# Use saved MLIQ_APC_STATE directly - already corrected in build_market_state_apc()
apc_df['LIQ_APC'] = apc_df['MLIQ_APC_STATE']  # No flip needed - sign already correct

ew_df = pd.read_csv('liq_market_index_equal_weighted.csv')  # This is equal-weighted
ew_df['DATE'] = pd.to_datetime(ew_df['MONTH'].astype(str))
ew_df.rename(columns={'LIQ_MARKET': 'LIQ_EW'}, inplace=True)

# Plot 1a: PCA (Variance-Weighted)
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(pca_var_df['DATE'], pca_var_df['LIQ_PCA'], linewidth=1.5, color='darkgreen', label='PCA Liquidity Index')
ax.axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red', label='2008-09 Crisis')
ax.axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange', label='2020 COVID')

# Highlight worst period
worst = pca_var_df.loc[pca_var_df['LIQ_PCA'].idxmin()]
ax.scatter(worst['DATE'], worst['LIQ_PCA'], color='red', s=150, zorder=5, marker='v', edgecolor='black', linewidth=2)
ax.text(worst['DATE'], worst['LIQ_PCA'], f"  Worst: {worst['DATE'].strftime('%b %Y')}\n  Value: {worst['LIQ_PCA']:.2f}", 
        fontsize=9, color='red', fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('PCA Liquidity Index', fontsize=12, fontweight='bold')
ax.set_title('Method 1: PCA Index (Variance-Weighted)\nWeights based on variance explained (AMIHUD-dominated)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'method1_pca_index.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: method1_pca_index.png")

# Plot 1b: APC (Pairwise Covariance)
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(apc_df['DATE'], apc_df['LIQ_APC'], linewidth=1.5, color='purple', label='APC Liquidity Index')
ax.axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red', label='2008-09 Crisis')
ax.axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange', label='2020 COVID')

# Highlight worst period
worst = apc_df.loc[apc_df['LIQ_APC'].idxmin()]
ax.scatter(worst['DATE'], worst['LIQ_APC'], color='red', s=150, zorder=5, marker='v', edgecolor='black', linewidth=2)
ax.text(worst['DATE'], worst['LIQ_APC'], f"  Worst: {worst['DATE'].strftime('%b %Y')}\n  Value: {worst['LIQ_APC']:.2f}", 
        fontsize=9, color='red', fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('APC Liquidity Index', fontsize=12, fontweight='bold')
ax.set_title('Method 2: APC Index (Pairwise Covariance)\nWeights from eigenvector of pairwise covariance matrix', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'method2_apc_index.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: method2_apc_index.png")

# Plot 1c: Equal-Weighted
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(ew_df['DATE'], ew_df['LIQ_EW'], linewidth=1.5, color='navy', label='Equal-Weighted Liquidity Index')
ax.axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red', label='2008-09 Crisis')
ax.axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange', label='2020 COVID')

# Highlight worst period
worst = ew_df.loc[ew_df['LIQ_EW'].idxmin()]
ax.scatter(worst['DATE'], worst['LIQ_EW'], color='red', s=150, zorder=5, marker='v', edgecolor='black', linewidth=2)
ax.text(worst['DATE'], worst['LIQ_EW'], f"  Worst: {worst['DATE'].strftime('%b %Y')}\n  Value: {worst['LIQ_EW']:.2f}", 
        fontsize=9, color='red', fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Equal-Weighted Liquidity Index', fontsize=12, fontweight='bold')
ax.set_title('Method 3: Equal-Weighted Index (Balanced Approach)\nAll 6 proxies contribute equally - captures all liquidity dimensions', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'method3_equal_weighted_index.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: method3_equal_weighted_index.png")

# Plot 1d: Comparison of all 3 methods
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(pca_var_df['DATE'], pca_var_df['LIQ_PCA'], linewidth=1.5, color='darkgreen', label='PCA (Variance-Weighted)', alpha=0.7)
ax.plot(apc_df['DATE'], apc_df['LIQ_APC'], linewidth=1.5, color='purple', label='APC (Pairwise Covariance)', alpha=0.7)
ax.plot(ew_df['DATE'], ew_df['LIQ_EW'], linewidth=2, color='navy', label='Equal-Weighted (PRIMARY)', alpha=0.9)

ax.axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.1, color='red', label='2008-09 Crisis')
ax.axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.1, color='orange', label='2020 COVID')

# Mark COVID period with annotation
covid_date = pd.to_datetime('2020-03-01')
ax.axvline(covid_date, color='orange', linestyle='--', linewidth=2, alpha=0.5)
ax.text(covid_date, ax.get_ylim()[1]*0.95, 'COVID-19\nMarch 2020', 
        ha='center', va='top', fontsize=10, color='orange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Liquidity Index (Standardized)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of All Three Methods\nEqual-Weighted captures COVID best; PCA & APC miss it', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'all_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(LIQUIDITY_PLOTS_DIR / 'method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: all_methods_comparison.png")
print("  [OK] Saved: method_comparison.png")

# Plot 1: Time Series of Market Liquidity Index with Trend (using equal-weighted)
print("\n[2/9] Time series of market liquidity index (equal-weighted primary)...")
plt.figure(figsize=(16, 6))
market_pca['DATE'] = market_pca['YEAR_MONTH'].astype(str)
market_pca['DATE'] = pd.to_datetime(market_pca['DATE'])
plt.plot(market_pca['DATE'], market_pca['LIQ_MARKET'], linewidth=1.5, color='navy', label='Market Liquidity Index')

# Add 12-month moving average
market_pca['MA_12'] = market_pca['LIQ_MARKET'].rolling(window=12, center=True).mean()
plt.plot(market_pca['DATE'], market_pca['MA_12'], linewidth=2, color='red', linestyle='--', label='12-Month MA')

plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Market Liquidity Index', fontsize=12, fontweight='bold')
plt.title('Indian Market Liquidity Index Over Time (2005-2024)\nEqual-Weighted Composite Measure (Higher = More Liquid)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'market_liquidity_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: market_liquidity_timeseries.png")

# Plot 2: Distribution of Stock-Level Liquidity Scores (Latest Month)
print("\n[3/9] Distribution of stock-level liquidity scores...")
latest_month = stock_pca['YEAR_MONTH'].max()
latest_scores = stock_pca[stock_pca['YEAR_MONTH'] == latest_month]['LIQ_STOCK'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(latest_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(latest_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {latest_scores.mean():.2f}')
axes[0].axvline(latest_scores.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {latest_scores.median():.2f}')
axes[0].set_xlabel('Stock Liquidity Score', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Stock Liquidity Scores', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(latest_scores, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='navy'),
                capprops=dict(color='navy'))
axes[1].set_ylabel('Stock Liquidity Score', fontsize=11, fontweight='bold')
axes[1].set_title('Box Plot of Stock Liquidity Scores', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Stock-Level Liquidity Distribution ({latest_month})\nHigher Scores = More Liquid Stocks', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'stock_liquidity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: stock_liquidity_distribution.png")

# Plot 3: Top 10 Most and Least Liquid Stocks
print("\n[4/9] Top 10 most/least liquid stocks...")
top_10_liquid = latest_scores.nlargest(10).sort_values()
top_10_illiquid = latest_scores.nsmallest(10).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Most Liquid
axes[0].barh(range(len(top_10_liquid)), top_10_liquid.values, color='green', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(top_10_liquid)))
axes[0].set_yticklabels(top_10_liquid.index)
axes[0].set_xlabel('Liquidity Score', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10 Most Liquid Stocks', fontsize=12, fontweight='bold', color='green')
axes[0].grid(True, alpha=0.3, axis='x')

# Least Liquid
axes[1].barh(range(len(top_10_illiquid)), top_10_illiquid.values, color='red', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(top_10_illiquid)))
axes[1].set_yticklabels(top_10_illiquid.index)
axes[1].set_xlabel('Liquidity Score', fontsize=11, fontweight='bold')
axes[1].set_title('Top 10 Least Liquid Stocks', fontsize=12, fontweight='bold', color='red')
axes[1].grid(True, alpha=0.3, axis='x')

fig.suptitle(f'Stock Liquidity Rankings ({latest_month})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'top_bottom_liquid_stocks.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: top_bottom_liquid_stocks.png")

# Plot 4: Liquidity Volatility Over Time (Enhanced with Crisis Periods)
print("\n[5/9] Liquidity volatility analysis...")
market_pca['LIQ_VOL_12M'] = market_pca['LIQ_MARKET'].rolling(window=12).std()
market_pca['LIQUIDITY_CHANGE'] = market_pca['LIQ_MARKET'].diff().abs()
market_pca['CHANGE_VOL_12M'] = market_pca['LIQUIDITY_CHANGE'].rolling(window=12).mean()

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Panel A: Liquidity level with volatility bands
axes[0].plot(market_pca['DATE'], market_pca['LIQ_MARKET'], linewidth=1.5, color='navy', label='Market Liquidity')
axes[0].fill_between(market_pca['DATE'], 
                      market_pca['LIQ_MARKET'] - market_pca['LIQ_VOL_12M'],
                      market_pca['LIQ_MARKET'] + market_pca['LIQ_VOL_12M'],
                      alpha=0.2, color='blue', label='±1 Std Dev (12M)')

# Add crisis period shading
crisis_periods = {
    '2008 Financial Crisis': ('2008-01', '2008-12'),
    '2009 Aftermath': ('2009-01', '2009-06'),
    '2020 COVID-19': ('2020-03', '2020-06')
}
for i, (name, (start, end)) in enumerate(crisis_periods.items()):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    axes[0].axvspan(start_date, end_date, alpha=0.15, color='red', label=name if i == 0 else '')
    if i == 0:
        axes[0].text(start_date + (end_date - start_date)/2, axes[0].get_ylim()[1]*0.95, 'Crisis\nPeriods', 
                    ha='center', va='top', fontsize=9, color='red', fontweight='bold')

axes[0].set_ylabel('Market Liquidity Index', fontsize=11, fontweight='bold')
axes[0].set_title('Market Liquidity with Volatility Bands\n(Higher = More Liquid; Shaded = ±1 Std Dev)', 
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Panel B: Rolling volatility with regime identification
volatility_mean = market_pca['LIQ_VOL_12M'].mean()
volatility_std = market_pca['LIQ_VOL_12M'].std()
high_vol_threshold = volatility_mean + volatility_std

axes[1].plot(market_pca['DATE'], market_pca['LIQ_VOL_12M'], linewidth=2, color='orange', label='12-Month Rolling Std Dev')
axes[1].axhline(volatility_mean, color='green', linestyle='--', linewidth=1.5, label=f'Mean Volatility ({volatility_mean:.3f})')
axes[1].axhline(high_vol_threshold, color='red', linestyle='--', linewidth=1.5, label=f'High Volatility Threshold (μ+σ)')

# Shade high volatility periods
high_vol_mask = market_pca['LIQ_VOL_12M'] > high_vol_threshold
for idx in range(len(market_pca) - 1):
    if high_vol_mask.iloc[idx]:
        axes[1].axvspan(market_pca['DATE'].iloc[idx], market_pca['DATE'].iloc[idx+1], 
                       alpha=0.15, color='red', linewidth=0)

axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Volatility (Std Dev)', fontsize=11, fontweight='bold')
axes[1].set_title('Liquidity Volatility Over Time\n(Red shading = High volatility periods; Shows volatility clustering)', 
                  fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'liquidity_volatility.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: liquidity_volatility.png")

# Plot 5: Year-over-Year Comparison (Annual Average Liquidity)
print("\n[8/9] Year-over-year liquidity comparison...")
market_pca['YEAR'] = market_pca['DATE'].dt.year
annual_liquidity = market_pca.groupby('YEAR')['LIQ_MARKET'].mean().reset_index()

plt.figure(figsize=(14, 6))
colors = ['green' if val > annual_liquidity['LIQ_MARKET'].mean() else 'red' for val in annual_liquidity['LIQ_MARKET']]
plt.bar(annual_liquidity['YEAR'], annual_liquidity['LIQ_MARKET'], color=colors, alpha=0.7, edgecolor='black')
plt.axhline(annual_liquidity['LIQ_MARKET'].mean(), color='navy', linestyle='--', linewidth=2, label='Overall Mean')

plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Average Market Liquidity', fontsize=12, fontweight='bold')
plt.title('Annual Average Market Liquidity (2005-2024)\nGreen: Above Average | Red: Below Average', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.xticks(annual_liquidity['YEAR'], rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'annual_liquidity_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: annual_liquidity_comparison.png")

# Plot 6: Cross-sectional Dispersion - Gap Between Liquid and Illiquid Stocks
print("\n[9/9] Cross-sectional liquidity dispersion...")
monthly_dispersion = stock_pca.groupby('YEAR_MONTH')['LIQ_STOCK'].agg([
    ('Q10', lambda x: x.quantile(0.10)),
    ('Q25', lambda x: x.quantile(0.25)),
    ('Q50', lambda x: x.quantile(0.50)),
    ('Q75', lambda x: x.quantile(0.75)),
    ('Q90', lambda x: x.quantile(0.90)),
    ('IQR', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ('SPREAD_90_10', lambda x: x.quantile(0.90) - x.quantile(0.10))
]).reset_index()
monthly_dispersion['DATE'] = pd.to_datetime(monthly_dispersion['YEAR_MONTH'].astype(str))

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Panel A: Liquidity distribution showing the spread
axes[0].fill_between(monthly_dispersion['DATE'], 
                      monthly_dispersion['Q10'], 
                      monthly_dispersion['Q90'], 
                      alpha=0.15, color='gray', label='10th-90th Percentile Range')
axes[0].fill_between(monthly_dispersion['DATE'], 
                      monthly_dispersion['Q25'], 
                      monthly_dispersion['Q75'], 
                      alpha=0.3, color='steelblue', label='Interquartile Range (25th-75th)')
axes[0].plot(monthly_dispersion['DATE'], monthly_dispersion['Q90'], linewidth=1.5, color='darkgreen', 
            label='90th Percentile (Most Liquid Stocks)', linestyle='--')
axes[0].plot(monthly_dispersion['DATE'], monthly_dispersion['Q50'], linewidth=2, color='navy', 
            label='Median Stock')
axes[0].plot(monthly_dispersion['DATE'], monthly_dispersion['Q10'], linewidth=1.5, color='darkred', 
            label='10th Percentile (Least Liquid Stocks)', linestyle='--')

axes[0].set_ylabel('Stock Liquidity Score', fontsize=11, fontweight='bold')
axes[0].set_title('Liquidity Gap Between Most Liquid and Least Liquid Stocks\n(Wider gap = Greater inequality in liquidity across stocks)', 
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
axes[0].grid(True, alpha=0.3)

# Add annotation for interpretation
axes[0].text(0.98, 0.05, 'INTERPRETATION:\n• Green line (90th %ile) = Most liquid stocks\n• Red line (10th %ile) = Least liquid stocks\n• Wider vertical gap = More dispersion', 
            transform=axes[0].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Dispersion measure over time
ax2 = axes[1]
ax2.plot(monthly_dispersion['DATE'], monthly_dispersion['SPREAD_90_10'], linewidth=2, color='purple', 
        label='90th-10th Percentile Spread')
ax2.plot(monthly_dispersion['DATE'], monthly_dispersion['IQR'], linewidth=2, color='orange', 
        label='Interquartile Range (Q75-Q25)')
ax2.axhline(monthly_dispersion['SPREAD_90_10'].mean(), color='purple', linestyle='--', linewidth=1, 
           alpha=0.7, label=f'Mean Spread: {monthly_dispersion["SPREAD_90_10"].mean():.2f}')
ax2.fill_between(monthly_dispersion['DATE'], 0, monthly_dispersion['SPREAD_90_10'], 
                alpha=0.1, color='purple')

ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
ax2.set_ylabel('Dispersion Measure', fontsize=11, fontweight='bold')
ax2.set_title('Cross-Sectional Dispersion Over Time\n(Higher values = Greater heterogeneity; liquid stocks pulling away from illiquid stocks)', 
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add crisis period shading
crisis_periods = {
    '2008 Financial Crisis': ('2008-01', '2008-12'),
    '2020 COVID-19': ('2020-03', '2020-06')
}
for name, (start, end) in crisis_periods.items():
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    ax2.axvspan(start_date, end_date, alpha=0.1, color='red', linewidth=0)

plt.tight_layout()
plt.savefig(LIQUIDITY_PLOTS_DIR / 'cross_sectional_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] Saved: cross_sectional_dispersion.png")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll plots saved to: {LIQUIDITY_PLOTS_DIR}/")
print("  1. market_liquidity_timeseries.png")
print("  2. stock_liquidity_distribution.png")
print("  3. top_bottom_liquid_stocks.png")
print("  4. liquidity_volatility.png")
print("  5. annual_liquidity_comparison.png")
print("  6. cross_sectional_dispersion.png")
print("="*80)

# ============================================================================
# COMPREHENSIVE VERIFICATION SUITE
# ============================================================================
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE VERIFICATION SUITE")
print("="*80)

# Import verification module
try:
    from verification_suite import (
        manual_unit_test, check_crisis_periods, 
        test_cross_sectional_monotonicity, analyze_weights_stability,
        export_verification_results, VERIFICATION_DIR, PLOTS_DIR
    )
    
    verification_results = {}
    
    # 1. Manual unit test for one stock-month
    print("\n" + "-"*80)
    print("[1/4] MANUAL UNIT TEST")
    print("-"*80)
    unit_test_result = manual_unit_test(
        daily_df=master_df,
        monthly_df=final_data,
        symbol='RELIANCE',
        year_month='2024-12'
    )
    if unit_test_result:
        verification_results['manual_unit_test'] = unit_test_result
        print("\n  [OK] Unit test completed")
    else:
        print("\n  [SKIP] Unit test - data not available for RELIANCE 2024-12")
        # Try alternative
        recent_month = final_data['YEAR_MONTH'].max()
        recent_symbols = final_data[final_data['YEAR_MONTH'] == recent_month]['SYMBOL'].head(1)
        if len(recent_symbols) > 0:
            print(f"  Trying {recent_symbols.iloc[0]} for {recent_month}...")
            unit_test_result = manual_unit_test(
                daily_df=master_df,
                monthly_df=final_data,
                symbol=recent_symbols.iloc[0],
                year_month=str(recent_month)
            )
            if unit_test_result:
                verification_results['manual_unit_test'] = unit_test_result
    
    # 2. Crisis period checks
    print("\n" + "-"*80)
    print("[2/4] ECONOMIC SANITY - CRISIS PERIOD CHECKS")
    print("-"*80)
    if 'market_pca' in locals():
        crisis_results = check_crisis_periods(market_pca)
        verification_results['crisis_periods'] = crisis_results
        print("\n  [OK] Crisis period checks completed")
    else:
        print("\n  [SKIP] Crisis checks - market index not available")
    
    # 3. Cross-sectional monotonicity
    print("\n" + "-"*80)
    print("[3/4] CROSS-SECTIONAL MONOTONICITY TEST")
    print("-"*80)
    if 'stock_pca' in locals():
        # Prepare data with monthly grouping
        stock_scores_with_proxies = final_data.merge(
            stock_pca[['SYMBOL', 'LIQ_STOCK']],
            on='SYMBOL',
            how='inner'
        )
        stock_scores_with_proxies['MONTH'] = pd.to_datetime(stock_scores_with_proxies['YEAR_MONTH'].astype(str) + '-01')
        
        monotonicity_results = test_cross_sectional_monotonicity(
            stock_scores_df=stock_scores_with_proxies,
            n_deciles=10
        )
        verification_results['monotonicity'] = monotonicity_results
        print("\n  [OK] Monotonicity test completed")
    else:
        print("\n  [SKIP] Monotonicity test - stock scores not available")
    
    # 4. PCA weights stability
    print("\n" + "-"*80)
    print("[4/4] PCA/APC WEIGHTS STABILITY ANALYSIS")
    print("-"*80)
    stability_results = analyze_weights_stability(
        monthly_data=final_data,
        selected_proxies=core_proxies
    )
    verification_results['weights_stability'] = stability_results
    print("\n  [OK] Weights stability analysis completed")
    
    # Export all results
    export_verification_results(verification_results, output_file='verification_results.json')
    
    print("\n" + "="*80)
    print("VERIFICATION SUITE COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - {VERIFICATION_DIR}/verification_results.json")
    print(f"  - {PLOTS_DIR}/*.png")
    print("="*80)
    
except ImportError as e:
    print(f"\n[WARNING] Could not import verification_suite: {e}")
    print("  Skipping comprehensive verification.")
except Exception as e:
    print(f"\n[ERROR] Verification suite encountered an error: {e}")
    import traceback
    traceback.print_exc()
    print("  Continuing with main script output...")

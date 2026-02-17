"""
LIQUIDITY PROXY VERIFICATION SUITE
===================================

This module provides comprehensive verification and validation for liquidity proxy calculations:
1. Hand-check unit tests (manual recomputation)
2. Economic sanity checks (crisis periods)
3. Cross-sectional monotonicity tests
4. PCA/APC weights stability analysis

Author: Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create verification results folder
VERIFICATION_DIR = Path("verification_results")
VERIFICATION_DIR.mkdir(exist_ok=True)
PLOTS_DIR = VERIFICATION_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("LIQUIDITY PROXY VERIFICATION SUITE")
print("="*80)

verification_results = {}

# =============================================================================
# 1. UNIT TEST: Manual Recomputation for One Stock-Month
# =============================================================================
print("\n[VERIFICATION 1] Unit Test: Manual Recomputation")
print("-"*80)

def manual_unit_test(daily_df, monthly_df, symbol='RELIANCE', year_month='2024-12'):
    """
    Hand-check one stock-month by manually recomputing:
    - AMIHUD
    - AMIVEST
    - ZERO_RATIO
    - LOT
    - ROLL_SPREAD
    """
    print(f"\nTesting {symbol} for {year_month}")
    
    # Get daily data for this stock-month
    daily_data = daily_df[
        (daily_df['SYMBOL'] == symbol) & 
        (daily_df['YEAR_MONTH'] == year_month)
    ].copy()
    
    if len(daily_data) == 0:
        print(f"  [ERROR] No data found for {symbol} {year_month}")
        return None
    
    print(f"  Found {len(daily_data)} days of data")
    
    results = {'symbol': symbol, 'year_month': year_month, 'n_days': len(daily_data)}
    
    # 1. AMIHUD = mean(|ret| / VALUE in millions)
    valid_amihud = (daily_data['DAILY_RETURN'].notna()) & (daily_data['VALUE'] > 0)
    if valid_amihud.sum() > 0:
        amihud_daily = daily_data.loc[valid_amihud, 'DAILY_RETURN'].abs() / (daily_data.loc[valid_amihud, 'VALUE'] / 1e6)
        manual_amihud = amihud_daily.mean()
        results['AMIHUD_manual'] = manual_amihud
        print(f"    AMIHUD (manual): {manual_amihud:.6e}")
    
    # 2. AMIVEST = 1 / [sum(VALUE) / sum(|ret|)]
    active_mask = (daily_data['DAILY_RETURN'].abs() > 0) & (daily_data['VALUE'] > 0)
    if active_mask.sum() >= 15:
        total_value = daily_data.loc[active_mask, 'VALUE'].sum()
        total_abs_return = daily_data.loc[active_mask, 'DAILY_RETURN'].abs().sum()
        if total_abs_return > 0:
            amivest_liq = total_value / total_abs_return
            manual_amivest = 1 / (amivest_liq + 1e-12)
            results['AMIVEST_manual'] = manual_amivest
            print(f"    AMIVEST (manual): {manual_amivest:.6e}")
    
    # 3. ZERO_RATIO = fraction of exact zeros
    valid_returns = daily_data['DAILY_RETURN'].notna()
    if valid_returns.sum() > 0:
        n_zeros = (daily_data.loc[valid_returns, 'DAILY_RETURN'] == 0).sum()
        manual_zero_ratio = n_zeros / valid_returns.sum()
        results['ZERO_RATIO_manual'] = manual_zero_ratio
        print(f"    ZERO_RATIO (manual): {manual_zero_ratio:.4f}")
    
    # 4. LOT = fraction with |ret| <= 5bp
    valid_lot = (daily_data['VALUE'] > 0) & (daily_data['DAILY_RETURN'].notna())
    if valid_lot.sum() > 0:
        n_near_zero = (daily_data.loc[valid_lot, 'DAILY_RETURN'].abs() <= 0.0005).sum()
        manual_lot = n_near_zero / valid_lot.sum()
        results['LOT_manual'] = manual_lot
        print(f"    LOT (manual): {manual_lot:.4f}")
    
    # 5. ROLL_SPREAD = 2 * sqrt(-cov(ΔP, ΔP_lag1))
    closes = daily_data['CLOSE'].values
    if len(closes) >= 10:
        price_changes = np.diff(closes)
        if len(price_changes) > 1:
            cov_val = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            if cov_val < 0:
                manual_roll = 2 * np.sqrt(-cov_val)
                results['ROLL_SPREAD_manual'] = manual_roll
                print(f"    ROLL_SPREAD (manual): {manual_roll:.6f}")
            else:
                results['ROLL_SPREAD_manual'] = 0.0
                print(f"    ROLL_SPREAD (manual): 0.0 (positive covariance)")
    
    # Get computed values from monthly data
    monthly_vals = monthly_df[
        (monthly_df['SYMBOL'] == symbol) & 
        (monthly_df['YEAR_MONTH'] == year_month)
    ]
    
    if len(monthly_vals) > 0:
        monthly_vals = monthly_vals.iloc[0]
        for proxy in ['AMIHUD', 'AMIVEST', 'ZERO_RATIO', 'LOT', 'ROLL_SPREAD']:
            if proxy in monthly_vals and f'{proxy}_manual' in results:
                computed = monthly_vals[proxy]
                manual = results[f'{proxy}_manual']
                diff = abs(computed - manual)
                rel_diff = diff / (abs(manual) + 1e-10)
                results[f'{proxy}_computed'] = computed
                results[f'{proxy}_diff'] = diff
                results[f'{proxy}_rel_diff'] = rel_diff
                
                status = "PASS" if rel_diff < 0.01 else "CHECK"
                print(f"    {proxy}: computed={computed:.6e}, diff={diff:.6e}, rel_diff={rel_diff:.2%} [{status}]")
    
    return results

# =============================================================================
# 2. ECONOMIC SANITY: Crisis Period Checks
# =============================================================================
print("\n[VERIFICATION 2] Economic Sanity: Crisis Period Checks")
print("-"*80)

def check_crisis_periods(market_index_df):
    """
    Verify market liquidity index spikes during:
    - 2008-09/10 (Lehman Brothers collapse)
    - 2020-03/04 (COVID-19 pandemic)
    """
    crisis_periods = {
        '2008_financial_crisis': ['2008-09', '2008-10', '2008-11', '2008-12'],
        '2009_crisis_aftermath': ['2009-01', '2009-02', '2009-03'],
        '2020_covid_shock': ['2020-03', '2020-04', '2020-05']
    }
    
    results = {}
    
    # Detect column name (could be LIQ_MARKET, MLIQ_MARKET, or other)
    liq_col = None
    for col in ['MLIQ_MARKET', 'LIQ_MARKET', 'MLIQ_PCA', 'LIQ_PCA']:
        if col in market_index_df.columns:
            liq_col = col
            break
    
    if liq_col is None:
        raise ValueError(f"No liquidity column found in market index. Available columns: {market_index_df.columns.tolist()}")
    
    # Convert MONTH to datetime if needed
    if 'MONTH' in market_index_df.columns:
        market_index_df['MONTH_STR'] = market_index_df['MONTH'].astype(str)
    elif 'YEAR_MONTH' in market_index_df.columns:
        market_index_df['MONTH_STR'] = market_index_df['YEAR_MONTH'].astype(str)
    
    # Get overall statistics
    liq_values = market_index_df[liq_col].values
    overall_mean = np.mean(liq_values)
    overall_std = np.std(liq_values)
    
    print(f"\n  Overall market liquidity: mean={overall_mean:.4f}, std={overall_std:.4f}")
    
    for crisis_name, months in crisis_periods.items():
        crisis_data = market_index_df[market_index_df['MONTH_STR'].isin(months)]
        
        if len(crisis_data) > 0:
            crisis_mean = crisis_data[liq_col].mean()
            crisis_max = crisis_data[liq_col].max()
            z_score = (crisis_mean - overall_mean) / (overall_std + 1e-10)
            
            results[crisis_name] = {
                'mean': crisis_mean,
                'max': crisis_max,
                'z_score': z_score,
                'months': months,
                'n_months': len(crisis_data)
            }
            
            status = "SPIKE DETECTED" if z_score > 1.0 else "Normal"
            print(f"\n  {crisis_name}:")
            print(f"    Mean: {crisis_mean:.4f} (z={z_score:.2f}) [{status}]")
            print(f"    Max: {crisis_max:.4f}")
            print(f"    Months: {', '.join(months[:3])}...")
    
    # Create crisis plot
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(market_index_df)), market_index_df[liq_col].values, label='Market Liquidity', linewidth=0.8)
    plt.axhline(overall_mean, color='gray', linestyle='--', alpha=0.5, label='Mean')
    plt.axhline(overall_mean + 2*overall_std, color='red', linestyle='--', alpha=0.5, label='+2σ')
    
    # Highlight crisis periods
    for crisis_name, months in crisis_periods.items():
        crisis_indices = market_index_df[market_index_df['MONTH_STR'].isin(months)].index
        if len(crisis_indices) > 0:
            plt.axvspan(crisis_indices.min(), crisis_indices.max(), alpha=0.2, label=crisis_name)
    
    plt.xlabel('Month Index')
    plt.ylabel('Market Liquidity Index')
    plt.title('Market Liquidity Over Time with Crisis Periods Highlighted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'crisis_periods_check.png', dpi=300)
    plt.close()
    
    print(f"\n  [SAVED] Crisis periods plot: {PLOTS_DIR / 'crisis_periods_check.png'}")
    
    return results

# =============================================================================
# 3. CROSS-SECTIONAL MONOTONICITY TEST
# =============================================================================
print("\n[VERIFICATION 3] Cross-Sectional Monotonicity Test")
print("-"*80)

def test_cross_sectional_monotonicity(stock_scores_df, n_deciles=10):
    """
    Each month, sort stocks into deciles by LIQ_STOCK score.
    Verify that mean AMIHUD and ZERO_RATIO increase across deciles.
    Should pass in >=80% of months.
    """
    print(f"\n  Testing monotonicity across {n_deciles} deciles...")
    
    monthly_results = []
    
    for month in sorted(stock_scores_df['MONTH'].unique()):
        month_data = stock_scores_df[stock_scores_df['MONTH'] == month].copy()
        
        if len(month_data) < n_deciles * 5:  # Need reasonable sample per decile
            continue
        
        # Assign deciles
        month_data['LIQ_DECILE'] = pd.qcut(month_data['LIQ_STOCK'], n_deciles, labels=False, duplicates='drop')
        
        # Check monotonicity for AMIHUD and ZERO_RATIO
        decile_stats = month_data.groupby('LIQ_DECILE').agg({
            'AMIHUD': 'mean',
            'ZERO_RATIO': 'mean'
        }).reset_index()
        
        # Test monotonicity (Spearman correlation should be close to 1)
        amihud_mono = stats.spearmanr(decile_stats['LIQ_DECILE'], decile_stats['AMIHUD'])[0]
        zero_mono = stats.spearmanr(decile_stats['LIQ_DECILE'], decile_stats['ZERO_RATIO'])[0]
        
        # Also check simple increasing pattern
        amihud_diffs = np.diff(decile_stats['AMIHUD'].values)
        zero_diffs = np.diff(decile_stats['ZERO_RATIO'].values)
        
        amihud_pct_increasing = (amihud_diffs > 0).sum() / len(amihud_diffs)
        zero_pct_increasing = (zero_diffs > 0).sum() / len(zero_diffs)
        
        monthly_results.append({
            'month': month,
            'amihud_spearman': amihud_mono,
            'zero_spearman': zero_mono,
            'amihud_pct_increasing': amihud_pct_increasing,
            'zero_pct_increasing': zero_pct_increasing,
            'amihud_passes': amihud_pct_increasing >= 0.7,
            'zero_passes': zero_pct_increasing >= 0.7
        })
    
    results_df = pd.DataFrame(monthly_results)
    
    # Summary statistics
    amihud_pass_rate = results_df['amihud_passes'].mean()
    zero_pass_rate = results_df['zero_passes'].mean()
    
    print(f"\n  Monotonicity Test Results:")
    print(f"    AMIHUD: {amihud_pass_rate:.1%} of months pass (target: >=80%)")
    print(f"    ZERO_RATIO: {zero_pass_rate:.1%} of months pass (target: >=80%)")
    print(f"\\n    Mean Spearman correlation:")
    print(f"      AMIHUD vs Decile: {results_df['amihud_spearman'].mean():.3f}")
    print(f"      ZERO_RATIO vs Decile: {results_df['zero_spearman'].mean():.3f}")
    
    overall_pass = (amihud_pass_rate >= 0.80) and (zero_pass_rate >= 0.80)
    status = "PASS" if overall_pass else "FAIL"
    print(f"\\n  Overall: [{status}]")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(results_df['amihud_pct_increasing'], bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(0.7, color='red', linestyle='--', label='70% threshold')
    axes[0].axvline(results_df['amihud_pct_increasing'].mean(), color='blue', linestyle='-', label='Mean')
    axes[0].set_xlabel('% of decile pairs increasing')
    axes[0].set_ylabel('# of months')
    axes[0].set_title('AMIHUD Monotonicity Across Months')
    axes[0].legend()
    
    axes[1].hist(results_df['zero_pct_increasing'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(0.7, color='red', linestyle='--', label='70% threshold')
    axes[1].axvline(results_df['zero_pct_increasing'].mean(), color='blue', linestyle='-', label='Mean')
    axes[1].set_xlabel('% of decile pairs increasing')
    axes[1].set_ylabel('# of months')
    axes[1].set_title('ZERO_RATIO Monotonicity Across Months')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'monotonicity_test.png', dpi=300)
    plt.close()
    
    print(f"\n  [SAVED] Monotonicity plot: {PLOTS_DIR / 'monotonicity_test.png'}")
    
    return {
        'amihud_pass_rate': amihud_pass_rate,
        'zero_pass_rate': zero_pass_rate,
        'overall_pass': overall_pass,
        'monthly_results': results_df.to_dict('records')
    }

# =============================================================================
# 4. PCA/APC WEIGHTS STABILITY ANALYSIS
# =============================================================================
print("\n[VERIFICATION 4] PCA/APC Weights Stability Across Subperiods")
print("-"*80)

def analyze_weights_stability(monthly_data, selected_proxies):
    """
    Compute PCA weights for subperiods and compare stability:
    - 2005-2012
    - 2013-2018
    - 2019-2024
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    subperiods = {
        '2005-2012': ('2005-01', '2012-12'),
        '2013-2018': ('2013-01', '2018-12'),
        '2019-2024': ('2019-01', '2024-12')
    }
    
    all_weights = {}
    
    for period_name, (start, end) in subperiods.items():
        period_data = monthly_data[
            (monthly_data['YEAR_MONTH'] >= start) & 
            (monthly_data['YEAR_MONTH'] <= end)
        ].copy()
        
        if len(period_data) < 100:
            print(f"\n  {period_name}: Insufficient data ({len(period_data)} obs)")
            continue
        
        # Standardize
        X = period_data[selected_proxies].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=1)
        pca.fit(X_scaled)
        
        weights = pca.components_[0]
        variance_explained = pca.explained_variance_ratio_[0]
        
        all_weights[period_name] = {
            'weights': dict(zip(selected_proxies, weights)),
            'variance_explained': variance_explained,
            'n_obs': len(period_data)
        }
        
        print(f"\n  {period_name}: {len(period_data)} obs, {variance_explained:.1%} variance explained")
        for proxy, weight in zip(selected_proxies, weights):
            print(f"    {proxy:<18}: {weight:>7.4f}")
    
    # Compare stability (correlation of weights across periods)
    periods = list(all_weights.keys())
    if len(periods) >= 2:
        print(f"\n  Weight Correlation Across Periods:")
        for i in range(len(periods)):
            for j in range(i+1, len(periods)):
                w1 = np.array([all_weights[periods[i]]['weights'][p] for p in selected_proxies])
                w2 = np.array([all_weights[periods[j]]['weights'][p] for p in selected_proxies])
                corr = np.corrcoef(w1, w2)[0, 1]
                print(f"    {periods[i]} vs {periods[j]}: {corr:.3f}")
    
    # Visualization
    if len(all_weights) > 0:
        weights_df = pd.DataFrame({
            period: [weights['weights'][p] for p in selected_proxies]
            for period, weights in all_weights.items()
        }, index=selected_proxies)
        
        plt.figure(figsize=(10, 6))
        weights_df.plot(kind='bar', width=0.8)
        plt.xlabel('Liquidity Proxy')
        plt.ylabel('PCA Weight')
        plt.title('PCA Weights Stability Across Subperiods')
        plt.legend(title='Period')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'weights_stability.png', dpi=300)
        plt.close()
        
        print(f"\n  [SAVED] Weights stability plot: {PLOTS_DIR / 'weights_stability.png'}")
    
    return all_weights

# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_verification_results(results_dict, output_file='verification_results.json'):
    """Export all verification results to JSON"""
    output_path = VERIFICATION_DIR / output_file
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj
    
    results_clean = convert_types(results_dict)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS EXPORTED")
    print(f"{'='*80}")
    print(f"  JSON: {output_path}")
    print(f"  Plots: {PLOTS_DIR}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    print("\nVerification suite module loaded successfully.")
    print("Import this module and call the verification functions with your data.")

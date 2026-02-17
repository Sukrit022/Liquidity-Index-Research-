"""
LIQUIDITY PROXY DEFINITIONS - SINGLE SOURCE OF TRUTH
====================================================

This module defines all liquidity proxies used in the analysis.
Each proxy has:
- name: Short identifier
- formula: Mathematical formula (LaTeX-style)
- description: Economic interpretation
- required_columns: List of daily data columns needed
- computation_level: 'daily' or 'monthly'
- direction: 'illiquidity' or 'liquidity'
- units: Measurement units
- reference: Academic citation
- implementation_notes: Critical implementation details
- dimension: Liquidity aspect captured

Convention: ALL PROXIES ARE STORED IN ILLIQUIDITY DIRECTION
(Higher value = MORE illiquid, less liquid)
"""

PROXY_DEFINITIONS = {
    'AMIHUD': {
        'name': 'Amihud (2002) Illiquidity Ratio',
        'formula': 'AMIHUD = |Return| / (Value / 10^6)',
        'description': 'Price impact per million rupees of trading volume',
        'required_columns': ['DAILY_RETURN', 'VALUE'],
        'computation_level': 'daily',
        'aggregation': 'mean',  # How to aggregate daily to monthly
        'direction': 'illiquidity',
        'units': 'basis points per million rupees',
        'reference': 'Amihud, Y. (2002). Illiquidity and stock returns. Journal of Financial Markets, 5(1), 31-56.',
        'implementation_notes': 'Use absolute return. Scale Value by 1e6 to avoid numerical issues.',
        'dimension': 'price_impact',
        'priority': 3
    },
    
    'HL_RANGE': {
        'name': 'High-Low Range (Normalized Daily Range)',
        'formula': 'HL_RANGE = 2 * (High - Low) / (High + Low)',
        'description': 'Normalized intraday price range as volatility/price uncertainty proxy',
        'required_columns': ['HIGH', 'LOW'],
        'computation_level': 'daily',
        'aggregation': 'mean',
        'direction': 'illiquidity',
        'units': 'fraction (0 to 1)',
        'reference': 'Parkinson (1980) range-based volatility estimator; NOT a spread estimator',
        'implementation_notes': """
        CRITICAL: This is a RANGE/VOLATILITY proxy, NOT a spread estimator.
        It measures intraday price uncertainty/volatility, not transaction costs.
        
        True spread estimators (like Corwin-Schultz 2012) use 2-day high/low ratios:
            β = Σ[log(H_t/L_t)^2] over 2 days
            α = (√(2β) - √β) / (3 - 2√2) - √(β / (3 - 2√2))
            spread = 2*(e^α - 1) / (1 + e^α)
        
        We use simplified normalized range: 2*(H-L)/(H+L)
        This captures price volatility/information flow, not bid-ask spreads.
        Classify as 'range_volatility' dimension, NOT 'spread_cost'.
        """,
        'dimension': 'range_volatility',
        'priority': 3
    },
    
    'ROLL_SPREAD': {
        'name': 'Roll (1984) Effective Spread',
        'formula': 'ROLL = 2 * sqrt(-Cov(ΔP_t, ΔP_t-1))',
        'description': 'Effective bid-ask spread from serial covariance of price changes',
        'required_columns': ['CLOSE'],
        'computation_level': 'monthly',
        'aggregation': None,  # Computed directly at monthly level
        'direction': 'illiquidity',
        'units': 'rupees',
        'reference': 'Roll, R. (1984). A simple implicit measure of the effective bid‐ask spread. Journal of Finance, 39(4), 1127-1139.',
        'implementation_notes': 'Set to 0 if covariance is positive (no bid-ask bounce). Requires ≥10 observations.',
        'dimension': 'spread_cost',
        'priority': 2
    },
    
    'ZERO_RATIO': {
        'name': 'Zero Return Ratio',
        'formula': 'ZERO_RATIO = (# days with Return = 0) / (# trading days)',
        'description': 'Proportion of trading days with exactly zero returns',
        'required_columns': ['DAILY_RETURN', 'IS_TRADING_DAY'],
        'computation_level': 'monthly',
        'aggregation': None,
        'direction': 'illiquidity',
        'units': 'proportion (0 to 1)',
        'reference': 'Lesmond, D., Ogden, J., & Trzcinka, C. (1999). A new estimate of transaction costs. Review of Financial Studies, 12(5), 1113-1141.',
        'implementation_notes': 'Count EXACT zeros only. Denominator = trading days with valid returns.',
        'dimension': 'speed_stickiness',
        'priority': 3
    },
    
    'NEAR_ZERO': {
        'name': 'Near-Zero Return Ratio (Tick-Size Robust)',
        'formula': 'NEAR_ZERO = (# days with |Return| ≤ 1bp) / (# trading days)',
        'description': 'Proportion of days with negligible price movement (≤1 basis point)',
        'required_columns': ['DAILY_RETURN', 'IS_TRADING_DAY'],
        'computation_level': 'monthly',
        'aggregation': None,
        'direction': 'illiquidity',
        'units': 'proportion (0 to 1)',
        'reference': 'Tick-size robust variant of Lesmond et al. (1999) for Indian markets',
        'implementation_notes': 'Threshold = 0.0001 (1bp). Accounts for NSE tick sizes (₹0.05, ₹0.10).',
        'dimension': 'speed_stickiness',
        'priority': 2
    },
    
    'FHT': {
        'name': 'Fong-Holden-Trzcinka (2017) Spread Estimator',
        'formula': 'FHT = 2 * σ * Φ^(-1)((1 + z)/2)',
        'description': 'Effective spread estimate from zero-return frequency and return volatility',
        'required_columns': ['DAILY_RETURN'],
        'computation_level': 'monthly',
        'aggregation': None,
        'direction': 'illiquidity',
        'units': 'percentage (spread as fraction of price)',
        'reference': 'Fong, K., Holden, C., & Trzcinka, C. (2017). What are the best liquidity proxies for global research? Review of Finance, 21(4), 1355-1401.',
        'implementation_notes': """
        CRITICAL FORMULA COMPONENTS:
        - z = proportion of zero returns among trading days (not including NTD)
        - σ = standard deviation of NON-zero returns
        - Φ^(-1) = inverse normal CDF
        
        Special cases:
        - If z = 0 (no zero returns): Set FHT = 0 (perfectly liquid)
        - If z ≥ 0.95: Set FHT = NaN (too degenerate)
        
        NTD HANDLING:
        - If NTD observable: z = zeros / (trading days + NTD)
        - If NTD NOT observable: z = zeros / trading days only
        
        DEGENERACY CHECK:
        - Exclude if valid_ratio < 20% OR pct_zeros > 80%
        - Common in highly liquid markets (e.g., large-cap Indian stocks)
        
        THIS IS A SPREAD ESTIMATOR, NOT A ZERO-VOLUME RATIO.
        """,
        'dimension': 'speed_stickiness',
        'priority': 2
    },
    
    'PASTOR': {
        'name': 'Pastor-Stambaugh (2003) Gamma',
        'formula': "r'_{t+1} = θ + φ*r'_t + γ*sign(r'_t)*volume_t + ε; PASTOR = -γ",
        'description': 'Return reversal coefficient capturing volume-related price impact',
        'required_columns': ['DAILY_RETURN', 'MARKET_RETURN', 'VALUE', 'VOLUME'],
        'computation_level': 'monthly',
        'aggregation': None,
        'direction': 'illiquidity',
        'units': 'regression coefficient (dimensionless)',
        'reference': 'Pastor, L., & Stambaugh, R. (2003). Liquidity risk and expected stock returns. Journal of Political Economy, 111(3), 642-685.',
        'implementation_notes': """
        Regression model:
        r'_{i,t+1} = θ + φ*r'_{i,t} + γ*sign(r'_{i,t})*volume_{i,t} + ε_{i,t+1}
        
        Where:
        - r'_{i,t} = excess return = stock return - equal-weighted market return
        - sign(r'_{i,t}) = +1, 0, or -1
        - volume_{i,t} = we use VALUE (rupee volume) instead of share volume
        - γ = price reversal coefficient (more negative = more illiquid)
        
        DIRECTION: Report -γ so higher value = more illiquid
        Require ≥8 observations per month.
        """,
        'dimension': 'price_impact',
        'priority': 1
    },
    
    'AMIVEST': {
        'name': 'Amivest Liquidity Ratio (Inverted)',
        'formula': 'AMIVEST_ILLIQ = 1 / (sum(VALUE) / sum(|Return|) + ε)',
        'description': 'Inverted Amivest: measures volume needed per unit price movement (higher = more illiquid)',
        'required_columns': ['DAILY_RETURN', 'VALUE'],
        'computation_level': 'monthly',
        'aggregation': 'sum_ratio_then_invert',
        'direction': 'illiquidity',
        'units': 'inverse rupees per return unit',
        'reference': 'Amihud, Y., & Mendelson, H. (1989). The effects of beta, bid-ask spread, residual risk, and size on stock returns. Journal of Finance, 44(2), 479-486.',
        'implementation_notes': """
        TRUE AMIVEST LIQUIDITY = sum(VALUE_d) / sum(|ret_d|) for active trading days
        Where active = |ret| > 0 AND VALUE > 0
        
        This differs from AMIHUD:
        - AMIHUD = mean(|ret| / VALUE) - averages RATIOS
        - AMIVEST = sum(VALUE) / sum(|ret|) - RATIO of SUMS
        
        Monthly computation:
        1. Sum VALUE across valid days
        2. Sum |Return| across same days
        3. AMIVEST_LIQ = total_value / total_return
        4. AMIVEST_ILLIQ = 1 / (AMIVEST_LIQ + 1e-12)
        5. Winsorize monthly values at 1%/99% (NOT daily)
        
        Requires ≥15 days with |ret| > 0 AND VALUE > 0.
        
        KEY DIFFERENCE: Not mechanically correlated with AMIHUD due to sum-then-divide vs divide-then-mean.
        """,
        'dimension': 'price_impact',
        'priority': 2
    },
    
    'LOT': {
        'name': 'Limited Order Turnover (LOT) - Threshold-Based',
        'formula': 'LOT = (# days with |Return| ≤ 5bp) / (# valid trading days)',
        'description': 'Proportion of days with no price movement beyond tick size (implicit transaction costs)',
        'required_columns': ['DAILY_RETURN', 'VALUE'],
        'computation_level': 'monthly',
        'aggregation': None,
        'direction': 'illiquidity',
        'units': 'proportion (0 to 1)',
        'reference': 'Lesmond, D., Ogden, J., & Trzcinka, C. (1999). Based on limited dependent variable approach.',
        'implementation_notes': """
        Threshold-based (NOT exact zeros):
        - LOT = fraction of days where |Return| ≤ 0.0005 (5 basis points)
        - Denominator = trading days with VALUE > 0 AND return not NaN
        - Threshold chosen to capture tick-constrained trading
        
        DIFFERS FROM ZERO_RATIO:
        - ZERO_RATIO: counts EXACT zeros only
        - LOT: counts returns within 5bp threshold
        
        Correlation with ZERO_RATIO should be 0.70-0.85 (not >0.98)
        """,
        'dimension': 'speed_stickiness',
        'priority': 1
    },
    
    'VOLUME_ILLIQ': {
        'name': 'Volume Illiquidity (Inverted)',
        'formula': 'VOLUME_ILLIQ = 1 / (1 + mean(Volume))',
        'description': 'Inverted average trading volume (higher = lower volume = more illiquid)',
        'required_columns': ['VOLUME'],
        'computation_level': 'monthly',
        'aggregation': 'mean_then_invert',
        'direction': 'illiquidity',
        'units': 'inverse shares',
        'reference': 'Standard volume-based liquidity proxy, inverted',
        'implementation_notes': """
        Inversion formula: 1/(1+x) maps [0,∞) → (0,1]
        Avoids division by zero.
        Higher value = lower volume = more illiquid.
        """,
        'dimension': 'activity_quantity',
        'priority': 2
    },
    
    'TURNOVER_ILLIQ': {
        'name': 'Turnover Illiquidity (Inverted)',
        'formula': 'TURNOVER_ILLIQ = 1 / (1 + mean(Value))',
        'description': 'Inverted rupee trading volume (higher = lower turnover = more illiquid)',
        'required_columns': ['VALUE'],
        'computation_level': 'monthly',
        'aggregation': 'mean_then_invert',
        'direction': 'illiquidity',
        'units': 'inverse rupees',
        'reference': 'Turnover-based liquidity proxy, inverted',
        'implementation_notes': """
        TRUE TURNOVER = Volume / Shares Outstanding
        WE USE: Value (rupee volume) as proxy (no shares outstanding data)
        Better name: "Dollar Volume Illiquidity"
        
        Inversion: 1/(1+x) to avoid division by zero.
        """,
        'dimension': 'activity_quantity',
        'priority': 3
    },
    
    'CV_TURNOVER': {
        'name': 'Coefficient of Variation of Turnover',
        'formula': 'CV_TURNOVER = std(Value) / mean(Value)',
        'description': 'Volatility of trading activity (higher CV = more erratic = more illiquid)',
        'required_columns': ['VALUE'],
        'computation_level': 'monthly',
        'aggregation': 'cv',
        'direction': 'illiquidity',
        'units': 'dimensionless (coefficient of variation)',
        'reference': 'Standard dispersion measure for trading activity',
        'implementation_notes': """
        Measures stability of trading activity.
        Higher CV = more erratic trading = less reliable liquidity.
        
        PREFERENCE: Level proxies (TURNOVER_ILLIQ, VOLUME_ILLIQ) preferred over dispersion.
        CV_TURNOVER is lower priority (priority=1) for activity dimension.
        """,
        'dimension': 'activity_quantity',
        'priority': 1
    },
    
    'DOLLAR_ILLIQ': {
        'name': 'Dollar Illiquidity (Normalized Price Impact)',
        'formula': 'DOLLAR_ILLIQ = |Return| / (Value / Close)',
        'description': 'Price impact normalized by share-equivalent volume',
        'required_columns': ['DAILY_RETURN', 'VALUE', 'CLOSE'],
        'computation_level': 'daily',
        'aggregation': 'mean',
        'direction': 'illiquidity',
        'units': 'basis points per share-equivalent',
        'reference': 'Variation of Amihud (2002) normalized by price',
        'implementation_notes': """
        Normalized version of Amihud:
        - Denominator = Value / Close ≈ number of shares traded
        - Add +1 to denominator to avoid division by zero
        
        Formula: |Return| / ((Value / Close) + 1)
        """,
        'dimension': 'price_impact',
        'priority': 2
    }
}


def validate_proxy_definition(proxy_name, definition):
    """Validate that a proxy definition has all required fields"""
    required_fields = [
        'name', 'formula', 'description', 'required_columns',
        'computation_level', 'direction', 'units', 'reference',
        'implementation_notes', 'dimension', 'priority'
    ]
    
    for field in required_fields:
        if field not in definition:
            raise ValueError(f"Proxy '{proxy_name}' missing required field: {field}")
    
    # Validate direction
    if definition['direction'] not in ['illiquidity', 'liquidity']:
        raise ValueError(f"Proxy '{proxy_name}' has invalid direction: {definition['direction']}")
    
    # Validate computation level
    valid_levels = ['daily', 'monthly', 'daily_then_monthly']
    if definition['computation_level'] not in valid_levels:
        raise ValueError(f"Proxy '{proxy_name}' has invalid computation_level: {definition['computation_level']}")
    
    # Validate priority
    if not isinstance(definition['priority'], int) or definition['priority'] < 1 or definition['priority'] > 3:
        raise ValueError(f"Proxy '{proxy_name}' has invalid priority: {definition['priority']} (must be 1-3)")
    
    return True


def get_proxy_list():
    """Get list of all defined proxies"""
    return list(PROXY_DEFINITIONS.keys())


def get_proxy_by_dimension(dimension):
    """Get all proxies for a specific dimension"""
    return [name for name, defn in PROXY_DEFINITIONS.items() 
            if defn['dimension'] == dimension]


def generate_definition_report(output_path='proxy_definitions_report.csv'):
    """Generate CSV report of all proxy definitions"""
    import pandas as pd
    
    rows = []
    for proxy_name, defn in PROXY_DEFINITIONS.items():
        rows.append({
            'Proxy': proxy_name,
            'Full Name': defn['name'],
            'Formula': defn['formula'],
            'Description': defn['description'],
            'Required Columns': ', '.join(defn['required_columns']),
            'Computation Level': defn['computation_level'],
            'Aggregation': defn.get('aggregation', 'N/A'),
            'Direction': defn['direction'],
            'Units': defn['units'],
            'Dimension': defn['dimension'],
            'Priority': defn['priority'],
            'Reference': defn['reference'],
            'Implementation Notes': defn['implementation_notes'].replace('\n', ' ').strip()
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# Validate all definitions on import
for proxy_name, definition in PROXY_DEFINITIONS.items():
    validate_proxy_definition(proxy_name, definition)

print(f"[OK] Loaded {len(PROXY_DEFINITIONS)} proxy definitions successfully")

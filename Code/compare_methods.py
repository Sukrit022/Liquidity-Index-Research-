"""Compare PCA vs Equal-Weighted vs APC for crisis detection"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all three indices
pca_ew = pd.read_csv('liq_market_index_pca.csv')  # Currently equal-weighted
pca_var = pd.read_csv('liq_market_index_pca_variance_weighted.csv')  # Pure PCA
apc = pd.read_csv('liq_market_index_apc_state.csv')

# Convert dates
pca_ew['DATE'] = pd.to_datetime(pca_ew['MONTH'].astype(str))
pca_var['DATE'] = pd.to_datetime(pca_var['MONTH'].astype(str))
apc['DATE'] = pd.to_datetime(apc['MONTH'].astype(str))

# LIQ_APC column is already sign-corrected in build_market_state_apc()
apc['LIQ'] = apc['LIQ_APC']

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Plot 1: Equal-Weighted
axes[0].plot(pca_ew['DATE'], pca_ew['LIQ_MARKET'], linewidth=1.5, color='navy')
axes[0].axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red', label='2008-09 Crisis')
axes[0].axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange', label='2020 COVID')
axes[0].set_ylabel('Liquidity Index', fontsize=11, fontweight='bold')
axes[0].set_title('Equal-Weighted Index (All proxies equal contribution)', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Highlight worst periods
worst_ew = pca_ew.nsmallest(1, 'LIQ_MARKET').iloc[0]
axes[0].scatter(worst_ew['DATE'], worst_ew['LIQ_MARKET'], color='red', s=100, zorder=5, marker='v')
axes[0].text(worst_ew['DATE'], worst_ew['LIQ_MARKET'], f"  Worst: {worst_ew['DATE'].strftime('%Y-%m')}", 
            fontsize=9, color='red', fontweight='bold', va='top')

# Plot 2: PCA (Variance-Weighted)
axes[1].plot(pca_var['DATE'], pca_var['LIQ_MARKET'], linewidth=1.5, color='darkgreen')
axes[1].axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red')
axes[1].axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange')
axes[1].set_ylabel('Liquidity Index', fontsize=11, fontweight='bold')
axes[1].set_title('PCA Index (Variance-weighted, dominated by AMIHUD)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

worst_pca = pca_var.nsmallest(1, 'LIQ_MARKET').iloc[0]
axes[1].scatter(worst_pca['DATE'], worst_pca['LIQ_MARKET'], color='red', s=100, zorder=5, marker='v')
axes[1].text(worst_pca['DATE'], worst_pca['LIQ_MARKET'], f"  Worst: {worst_pca['DATE'].strftime('%Y-%m')}", 
            fontsize=9, color='red', fontweight='bold', va='top')

# Plot 3: APC
axes[2].plot(apc['DATE'], apc['LIQ'], linewidth=1.5, color='purple')
axes[2].axvspan(pd.to_datetime('2008-01'), pd.to_datetime('2009-06'), alpha=0.15, color='red')
axes[2].axvspan(pd.to_datetime('2020-03'), pd.to_datetime('2020-06'), alpha=0.15, color='orange')
axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Liquidity Index', fontsize=11, fontweight='bold')
axes[2].set_title('APC Index (Pairwise covariance method)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

worst_apc = apc.nsmallest(1, 'LIQ').iloc[0]
axes[2].scatter(worst_apc['DATE'], worst_apc['LIQ'], color='red', s=100, zorder=5, marker='v')
axes[2].text(worst_apc['DATE'], worst_apc['LIQ'], f"  Worst: {worst_apc['DATE'].strftime('%Y-%m')}", 
            fontsize=9, color='red', fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('Liquidity_Plots/method_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: Liquidity_Plots/method_comparison.png")

# Print crisis detection comparison
print("\n" + "="*70)
print("CRISIS DETECTION COMPARISON")
print("="*70)

print("\n2020 COVID-19 (March 2020):")
covid_date = pd.to_datetime('2020-03-01')
print(f"  Equal-Weighted: {pca_ew[pca_ew['DATE']==covid_date]['LIQ_MARKET'].values[0]:.2f}")
print(f"  PCA (variance):  {pca_var[pca_var['DATE']==covid_date]['LIQ_MARKET'].values[0]:.2f}")
print(f"  APC:            {apc[apc['DATE']==covid_date]['LIQ'].values[0]:.2f}")

print("\n2008 Financial Crisis (October 2008):")
crisis_date = pd.to_datetime('2008-10-01')
print(f"  Equal-Weighted: {pca_ew[pca_ew['DATE']==crisis_date]['LIQ_MARKET'].values[0]:.2f}")
print(f"  PCA (variance):  {pca_var[pca_var['DATE']==crisis_date]['LIQ_MARKET'].values[0]:.2f}")
print(f"  APC:            {apc[apc['DATE']==crisis_date]['LIQ'].values[0]:.2f}")

print("\nWorst month overall:")
print(f"  Equal-Weighted: {worst_ew['DATE'].strftime('%Y-%m')} ({worst_ew['LIQ_MARKET']:.2f})")
print(f"  PCA (variance):  {worst_pca['DATE'].strftime('%Y-%m')} ({worst_pca['LIQ_MARKET']:.2f})")
print(f"  APC:            {worst_apc['DATE'].strftime('%Y-%m')} ({worst_apc['LIQ']:.2f})")

print("\n" + "="*70)
print("OBSERVATIONS:")
print("- Equal-Weighted: Identifies COVID period as largest liquidity deterioration")
print("- PCA (variance): Dominated by AMIHUD; identifies 2009 as worst period")
print("- APC: Similar to PCA; identifies 2009 as worst period")
print("\nNOTE: Whether 2008-09 or 2020 was the 'worse' liquidity event depends on")
print("which dimensions of liquidity are weighted most heavily. All three methods")
print("detect both events; they differ in relative severity ranking.")
print("="*70)

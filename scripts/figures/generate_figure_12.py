import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "jades_highz_physical.csv"
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "data" / "interim" / "jades_spec_highz.csv"
OUTPUT_PATH = PROJECT_ROOT / "site" / "public" / "figures" / "figure_12_simpson.png"
(OUTPUT_PATH.parent).mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Normalise column names: use z_best as fallback for z_spec, t_stellar_Gyr for mwa
if 'z_spec' not in df.columns or df['z_spec'].notna().sum() < 10:
    df['z_spec'] = df.get('z_best', df.get('z_phot'))
if 'mwa' not in df.columns:
    df['mwa'] = df.get('t_stellar_Gyr', df.get('age_ratio'))
if 'age_ratio' in df.columns and 'mwa' not in df.columns:
    df['mwa'] = df['age_ratio']

# Filter valid
df = df.dropna(subset=['log_Mstar', 'mwa', 'z_spec'])
df = df[(df['log_Mstar'] > 6) & (df['z_spec'] > 0) & (df['mwa'] > 0)]

# Calculate quantities
from astropy.cosmology import Planck18 as cosmo
log_Mh = stellar_to_halo_mass(df['log_Mstar'].values, df['z_spec'].values)
df['gamma_t'] = tep_gamma(log_Mh, df['z_spec'].values)

df['t_cosmic'] = cosmo.age(df['z_spec'].values).value
df['age_ratio'] = df['mwa'] / df['t_cosmic']

# Filter outliers for plotting clarity
df = df[df['age_ratio'] < 2.0]
df = df[df['gamma_t'] < 5.0]

# Define bins
bins = [4, 6, 8, 20]
labels = ['z = 4-6', 'z = 6-8', 'z > 8']
df['z_bin'] = pd.cut(df['z_spec'], bins=bins, labels=labels)

# Set style
set_pub_style(scale=1.0)
fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])

# Plot scatter points colored by bin
colors = {'z = 4-6': COLORS['accent'], 'z = 6-8': COLORS['highlight'], 'z > 8': COLORS['primary']}
for label in labels:
    subset = df[df['z_bin'] == label]
    ax.scatter(subset['gamma_t'], subset['age_ratio'], c=colors[label], label=label, alpha=0.6, s=50, edgecolor='w')
    
    # Bin-wise trend lines
    if len(subset) > 5:
        z = np.polyfit(subset['gamma_t'], subset['age_ratio'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['gamma_t'].min(), subset['gamma_t'].max(), 100)
        ax.plot(x_range, p(x_range), c=colors[label], linestyle='-', linewidth=2)

# Global trend line
z_global = np.polyfit(df['gamma_t'], df['age_ratio'], 1)
p_global = np.poly1d(z_global)
x_global = np.linspace(df['gamma_t'].min(), df['gamma_t'].max(), 100)
ax.plot(x_global, p_global(x_global), linestyle='--', linewidth=2.5, label='Global Trend', color=COLORS['text'])

# Annotations
rho_global, p_global_val = stats.spearmanr(df['gamma_t'], df['age_ratio'])
ax.text(0.05, 0.95, f'Global Correlation: ρ = {rho_global:.2f}', transform=ax.transAxes, 
        fontsize=12, fontweight='bold', verticalalignment='top')

# Formatting
ax.set_xlabel(r'Temporal Enhancement Factor ($\Gamma_t$)')
ax.set_ylabel(r'Age Ratio (Age / $t_{cosmic}$)')
ax.set_title("Simpson's Paradox in Spectroscopic Sample")
ax.legend(frameon=True, framealpha=0.9, facecolor=COLORS['background'])
ax.grid(True, alpha=0.3)

# Save
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Figure saved to {OUTPUT_PATH}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE
from scripts.utils.tep_model import compute_gamma_t as tep_gamma

DATA_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "site" / "public" / "figures"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def load_uncover_data():
    return pd.read_csv(DATA_PATH / "step_002_uncover_full_sample_tep.csv")

def plot_figure_1_tep_model():
    """Figure 1: The TEP Metric Coupling Gamma_t(M_h, z)."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    z_range = np.linspace(0, 12, 100)
    masses = [10, 11, 12, 13] # log Mh
    
    for log_mh in masses:
        gamma = tep_gamma(log_mh, z_range)
        
        label = rf"$\log M_h = {log_mh}$"
        if log_mh == 12:
            ax.plot(z_range, gamma, '--', color=COLORS['text'], label=label, alpha=0.5)
        elif log_mh > 12:
            ax.plot(z_range, gamma, label=label, color=COLORS['highlight'])
        else:
            ax.plot(z_range, gamma, label=label, color=COLORS['secondary'])
            
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"Chronological Enhancement $\Gamma_t$")
    ax.set_title(r"Figure 1: The TEP Metric Coupling")
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_1_tep_model.png")
    print(f"Saved Figure 1 to {OUTPUT_PATH / 'figure_1_tep_model.png'}")

def plot_figure_2_red_monsters():
    """Figure 2: Red Monsters SFE - Standard vs TEP."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    # Data from Table 3
    monsters = ['S1', 'S2', 'S3']
    sfe_obs = np.array([0.50, 0.50, 0.50])
    sfe_true = np.array([0.31, 0.35, 0.37])
    gamma = np.array([1.94, 1.64, 1.54])
    
    x = np.arange(len(monsters))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, sfe_obs, width, label='Standard SFE', color=COLORS['gray'], alpha=0.6)
    rects2 = ax.bar(x + width/2, sfe_true, width, label='TEP Corrected SFE', color=COLORS['primary'])
    
    # Theoretical Max line
    ax.axhline(0.20, color=COLORS['accent'], linestyle='--', linewidth=1.0, label='LambdaCDM Limit (~0.20)')
    
    ax.set_ylabel(r'Star Formation Efficiency ($\epsilon$)')
    ax.set_title('Figure 2: Red Monsters Efficiency Crisis')
    ax.set_xticks(x)
    ax.set_xticklabels(monsters)
    ax.legend()
    
    # Annotate Gamma
    for i, g in enumerate(gamma):
        ax.text(x[i] + width/2, sfe_true[i] + 0.01, rf"$\Gamma_t={g:.1f}$", ha='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_2_red_monsters.png")
    print(f"Saved Figure 2 to {OUTPUT_PATH / 'figure_2_red_monsters.png'}")

def plot_figure_3_inversion(df):
    """Figure 3: Mass-sSFR Inversion (Low-z vs High-z)."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    # Low z bin (4-6)
    mask_low = (df['z_phot'] >= 4) & (df['z_phot'] < 6)
    # High z bin (7-10)
    mask_high = (df['z_phot'] >= 7) & (df['z_phot'] < 10)

    # Calculate correlations
    rho_low, _ = spearmanr(df[mask_low]['log_Mstar'], df[mask_low]['log_ssfr'])
    rho_high, _ = spearmanr(df[mask_high]['log_Mstar'], df[mask_high]['log_ssfr'])
    
    # Bin by mass
    bins = np.linspace(8, 11, 7)
    df['mass_bin'] = pd.cut(df['log_Mstar'], bins)
    
    # Low Z
    grp_low = df[mask_low].groupby('mass_bin', observed=True)['log_ssfr'].agg(['mean', 'std', 'count']).reset_index()
    grp_low['err'] = grp_low['std'] / np.sqrt(grp_low['count'])
    
    # High Z
    grp_high = df[mask_high].groupby('mass_bin', observed=True)['log_ssfr'].agg(['mean', 'std', 'count']).reset_index()
    grp_high['err'] = grp_high['std'] / np.sqrt(grp_high['count'])
    
    x_low = [i.mid for i in grp_low['mass_bin']]
    x_high = [i.mid for i in grp_high['mass_bin']]
    
    label_low = f'Low-$z$ ($4<z<6$): Downsizing ($\\rho={rho_low:.2f}$)'
    label_high = f'High-$z$ ($z>7$): Inversion ($\\rho={rho_high:.2f}$)'
    
    ax.errorbar(x_low, grp_low['mean'], yerr=grp_low['err'], fmt='o-', label=label_low, color=COLORS['gray'])
    ax.errorbar(x_high, grp_high['mean'], yerr=grp_high['err'], fmt='s-', label=label_high, color=COLORS['primary'])
    
    ax.set_xlabel(r'Stellar Mass $\log(M_*/M_\odot)$')
    ax.set_ylabel(r'$\log(\mathrm{sSFR} / \mathrm{yr}^{-1})$')
    ax.set_title('Figure 3: The Downsizing Inversion')
    ax.legend()
    
    # Annotation for the transition
    ax.annotate(r'Transition at $z \approx 7$', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, fontweight='bold', color=COLORS['text'], ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_3_inversion.png")
    print(f"Saved Figure 3 to {OUTPUT_PATH / 'figure_3_inversion.png'}")

def plot_figure_4_gamma_age(df):
    """Figure 4: Gamma_t vs Age Ratio (Partial Correlation Visualization)."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    # Filter valid
    mask = (df['z_phot'] > 4) & (df['age_ratio'] > 0) & (df['age_ratio'] < 1.0)
    sub = df[mask]
    
    # Bin by Gamma_t
    bins = np.logspace(np.log10(0.1), np.log10(5), 10)
    sub['gamma_bin'] = pd.cut(sub['gamma_t'], bins)
    
    grp = sub.groupby('gamma_bin', observed=True)['age_ratio'].agg(['mean', 'std', 'count']).reset_index()
    grp['err'] = grp['std'] / np.sqrt(grp['count'])
    x = [i.mid for i in grp['gamma_bin']]
    
    ax.errorbar(x, grp['mean'], yerr=grp['err'], fmt='o-', color=COLORS['primary'], label='Observed Age Ratio')
    
    # TEP Prediction Line (qualitative)
    # Age_obs ~ Age_true * Gamma_t
    # Ratio ~ Gamma_t
    # Normalize to mean
    # ax.plot(x, 0.15 * np.array(x), '--', color=COLORS['accent'], label='TEP Prediction')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Enhancement Factor $\Gamma_t$')
    ax.set_ylabel('Age Ratio (Age / $t_{cosmic}$)')
    ax.set_title(r'Figure 4: Mass-Dependent Age Scaling')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_4_gamma_age.png")
    print(f"Saved Figure 4 to {OUTPUT_PATH / 'figure_4_gamma_age.png'}")

def plot_figure_5_dust_anomaly(df):
    """Figure 5: The z > 8 Dust Anomaly (The Mass-Dust Inversion)."""
    set_pub_style(scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE['full_width'])
    
    # Left: Low z (4-6)
    mask_low = (df['z_phot'] >= 4) & (df['z_phot'] < 6)
    sub_low = df[mask_low]
    
    # Right: High z (8-10)
    mask_high = (df['z_phot'] >= 8) & (df['z_phot'] < 10)
    sub_high = df[mask_high]
    
    # Scatter + Trend
    ax1.scatter(sub_low['log_Mstar'], sub_low['dust'], alpha=0.1, color=COLORS['gray'], s=5)
    
    # Trend line
    z1 = np.polyfit(sub_low['log_Mstar'], sub_low['dust'], 1)
    p1 = np.poly1d(z1)
    x_range = np.linspace(8, 11, 10)
    ax1.plot(x_range, p1(x_range), '--', color=COLORS['text'], label=f'Trend ($\\rho \\approx 0$)')
    
    ax1.set_title(r'(a) Low-$z$ ($4 < z < 6$): No Correlation')
    ax1.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax1.set_ylabel(r'Dust Attenuation $A_V$')
    ax1.set_ylim(0, 3)
    ax1.legend()
    
    # High Z
    im = ax2.scatter(sub_high['log_Mstar'], sub_high['dust'], c=sub_high['gamma_t'], 
                     cmap='PuBu', alpha=0.8, s=20, norm=plt.Normalize(0.1, 3))
    
    z2 = np.polyfit(sub_high['log_Mstar'], sub_high['dust'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_range, p2(x_range), '-', color=COLORS['highlight'], linewidth=1.0, label=r'Trend ($\rho = +0.56$)')
    
    ax2.set_title(r'(b) High-$z$ ($z > 8$): The Mass-Dust Inversion')
    ax2.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax2.set_ylim(0, 3)
    ax2.legend(loc='upper left')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(r'Enhancement $\Gamma_t$')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_5_dust_anomaly.png")
    print(f"Saved Figure 5 to {OUTPUT_PATH / 'figure_5_dust_anomaly.png'}")

def plot_figure_6_replication():
    """Figure 6: Cross-Survey Replication (Summary Plot)."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    surveys = ['UNCOVER', 'CEERS', 'COSMOS-Web']
    rhos = [0.56, 0.68, 0.63]
    cis = [[0.46, 0.65], [0.52, 0.80], [0.59, 0.67]]
    ns = [283, 82, 918]
    
    y = np.arange(len(surveys))
    
    # Plot correlations
    for i in range(len(surveys)):
        ax.errorbar(rhos[i], y[i], xerr=[[rhos[i]-cis[i][0]], [cis[i][1]-rhos[i]]], 
                    fmt='o', color=COLORS['primary'], markersize=8, capsize=5)
        ax.text(rhos[i], y[i]+0.15, f"N={ns[i]}", ha='center', fontsize=9)
        
    ax.set_yticks(y)
    ax.set_yticklabels(surveys)
    ax.axvline(0, color=COLORS['gray'], linestyle='--')
    ax.set_xlabel(r'Mass-Dust Correlation $\rho(M_*, A_V)$ at $z > 8$')
    ax.set_title(r'Figure 6: Independent Replication')
    ax.set_xlim(-0.2, 1.0)
    
    # Add Combined
    ax.axvspan(0.60, 0.64, color=COLORS['highlight'], alpha=0.2, label='Weighted Average')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_6_replication.png")
    print(f"Saved Figure 6 to {OUTPUT_PATH / 'figure_6_replication.png'}")

def plot_figure_7_regimes(df):
    """Figure 7: Regime Separation (Gamma < 1 vs Gamma > 1)."""
    set_pub_style(scale=1.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE['full_width'])
    
    mask = (df['z_phot'] > 8)
    sub = df[mask]
    
    enhanced = sub[sub['gamma_t'] > 1]
    suppressed = sub[sub['gamma_t'] < 1]
    
    # Plot Histograms of Dust
    ax.hist(suppressed['dust'], bins=20, density=True, alpha=0.6, color=COLORS['gray'], label=r'Suppressed ($\Gamma_t < 1$)')
    ax.hist(enhanced['dust'], bins=10, density=True, alpha=0.6, color=COLORS['primary'], label=r'Enhanced ($\Gamma_t > 1$)')
    
    ax.set_xlabel(r'Dust Attenuation $A_V$')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'Figure 7: Regime Separation at $z > 8$')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "figure_7_regimes.png")
    print(f"Saved Figure 7 to {OUTPUT_PATH / 'figure_7_regimes.png'}")

def main():
    print("Loading data...")
    df = load_uncover_data()
    
    print("Generating figures...")
    plot_figure_1_tep_model()
    plot_figure_2_red_monsters()
    plot_figure_3_inversion(df)
    plot_figure_4_gamma_age(df)
    plot_figure_5_dust_anomaly(df)
    plot_figure_6_replication()
    plot_figure_7_regimes(df)
    
    print("Done.")

if __name__ == "__main__":
    main()

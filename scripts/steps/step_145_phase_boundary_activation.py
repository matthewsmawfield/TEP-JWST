#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 5.7s.
"""
Step 145: AGB Dust Phase Boundary + Activation Curve + Spectroscopic Sharpening

Three complementary "aha" analyses:

1. AGB DUST PHASE BOUNDARY MAP
   Plot every galaxy in (M*, z) space colored by dust detection.
   Overlay the TEP-predicted t_eff = 0.3 Gyr isochrone — the predicted
   phase boundary between dust-producing and dust-free galaxies.
   A mass-only threshold would be a vertical line; TEP predicts a specific
   CURVE whose shape encodes both mass and redshift dependence.

2. ACTIVATION CURVE FIT
   The Gamma_t-dust correlation strength evolves dramatically with redshift:
   rho ~ 0 at z=4-6, rising to rho = 0.73 at z=9-10.
   Fit this pattern to competing functional forms:
   - TEP: rho(z) ~ alpha(z) ~ sqrt(1+z)
   - Linear: rho(z) ~ a + b*z
   - Step function: rho(z) = 0 for z<z_c, rho_max for z>=z_c
   - Constant
   If sqrt(1+z) wins by AIC, that's a prediction no mass proxy makes.

3. SPECTROSCOPIC SHARPENING TEST
   For the N=147 spec-z galaxies, recompute the primary correlations.
   If rho INCREASES with exact redshifts (vs photo-z), it rules out
   photo-z artifacts and directly addresses limitation #3.

Outputs:
- results/outputs/step_145_phase_boundary_activation.json
- results/figures/figure_145_phase_boundary.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import (
    compute_gamma_t, stellar_to_halo_mass_behroozi_like,
    compute_t_eff, KAPPA_GAL,  # Shared TEP model
)
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "145"  # Pipeline step number
STEP_NAME = "phase_boundary_activation"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products
RESULTS_INTERIM = PROJECT_ROOT / "results" / "interim"

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore")

AGB_THRESHOLD_GYR = 0.3  # AGB dust onset timescale
DUST_DETECTION_THRESHOLD = 0.1  # Av threshold for "dust detected"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_uncover():
    """Load UNCOVER with TEP properties."""
    for path in [
        RESULTS_INTERIM / "step_002_uncover_full_sample_tep.csv",
        RESULTS_INTERIM / "step_001_uncover_full_sample.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        return None

    z_col = "z_phot" if "z_phot" in df.columns else "z"
    df["z"] = df[z_col]
    return df


def load_specz():
    """Load spectroscopic sample."""
    # We need a sample with dust, z, and log_Mstar. CEERS has this.
    path = DATA_INTERIM / "ceers_highz_sample.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    
    # Normalize columns
    if "z_spec" in df.columns:
        df["z"] = df["z_spec"]
    else:
        df["z"] = df.get("z_best", df.get("z_phot"))
    
    if "dust" not in df.columns:
        if "A_V" in df.columns:
            df["dust"] = df["A_V"]
        elif "dust2" in df.columns:
            df["dust"] = df["dust2"]
        else:
            df["dust"] = np.nan

    # Compute TEP quantities
    valid = df["log_Mstar"].notna() & df["z"].notna()
    df.loc[valid, "log_mh"] = stellar_to_halo_mass_behroozi_like(
        df.loc[valid, "log_Mstar"].values, df.loc[valid, "z"].values
    )
    df.loc[valid, "gamma_t"] = compute_gamma_t(
        df.loc[valid, "log_mh"].values, df.loc[valid, "z"].values
    )
    df.loc[valid, "t_eff"] = compute_t_eff(
        df.loc[valid, "log_mh"].values, df.loc[valid, "z"].values
    )
    return df


# ── Test 1: AGB Dust Phase Boundary ─────────────────────────────────────────

def compute_agb_boundary(z_range=(4, 11), n_points=200):
    """
    Compute the TEP-predicted AGB dust onset boundary in (M*, z) space.

    For each z, find the stellar mass M* such that t_eff(M*, z) = 0.3 Gyr.
    """
    z_vals = np.linspace(z_range[0], z_range[1], n_points)
    mstar_boundary = np.full(n_points, np.nan)

    for i, z in enumerate(z_vals):
        # Search for M* where t_eff = AGB_THRESHOLD_GYR
        def objective(log_mstar):
            log_mh = stellar_to_halo_mass_behroozi_like(
                np.array([log_mstar]), np.array([z])
            )[0]
            t_eff = compute_t_eff(
                np.array([log_mh]), np.array([z])
            )[0]
            return (t_eff - AGB_THRESHOLD_GYR) ** 2

        # Grid search + refinement
        best_mstar = np.nan
        best_val = np.inf
        for m_try in np.linspace(6.0, 12.0, 60):
            val = objective(m_try)
            if val < best_val:
                best_val = val
                best_mstar = m_try

        if best_val < 0.01:
            try:
                res = optimize.minimize_scalar(
                    objective, bounds=(6.0, 12.0), method="bounded"
                )
                if res.fun < 0.001:
                    mstar_boundary[i] = res.x
            except Exception:
                mstar_boundary[i] = best_mstar

    return z_vals, mstar_boundary


def test1_phase_boundary(df):
    """
    Map dust detection vs t_eff threshold in (M*, z) space.
    """
    print_status("TEST 1: AGB Dust Phase Boundary Map")

    sub = df.dropna(subset=["log_Mstar", "z", "t_eff", "dust"]).copy()
    sub["dust_detected"] = sub["dust"] > DUST_DETECTION_THRESHOLD
    n = len(sub)

    # Split by AGB threshold
    above_teff = sub[sub["t_eff"] >= AGB_THRESHOLD_GYR]
    below_teff = sub[sub["t_eff"] < AGB_THRESHOLD_GYR]

    frac_above = above_teff["dust_detected"].mean() if len(above_teff) > 0 else 0
    frac_below = below_teff["dust_detected"].mean() if len(below_teff) > 0 else 0

    # Compute boundary curve
    z_boundary, mstar_boundary = compute_agb_boundary()

    # Classification accuracy
    predicted_dusty = sub["t_eff"] >= AGB_THRESHOLD_GYR
    actual_dusty = sub["dust_detected"]

    tp = ((predicted_dusty) & (actual_dusty)).sum()
    fp = ((predicted_dusty) & (~actual_dusty)).sum()
    fn = ((~predicted_dusty) & (actual_dusty)).sum()
    tn = ((~predicted_dusty) & (~actual_dusty)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / n if n > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # z>8 subsample
    z8 = sub[sub["z"] >= 8.0]
    above_z8 = z8[z8["t_eff"] >= AGB_THRESHOLD_GYR]
    below_z8 = z8[z8["t_eff"] < AGB_THRESHOLD_GYR]
    frac_above_z8 = above_z8["dust_detected"].mean() if len(above_z8) > 0 else 0
    frac_below_z8 = below_z8["dust_detected"].mean() if len(below_z8) > 0 else 0

    # Compare with mass-only threshold: find mass that gives same split ratio
    mass_thresh = sub["log_Mstar"].quantile(len(above_teff) / n)
    above_mass = sub[sub["log_Mstar"] >= mass_thresh]
    below_mass = sub[sub["log_Mstar"] < mass_thresh]
    frac_above_mass = above_mass["dust_detected"].mean() if len(above_mass) > 0 else 0
    frac_below_mass = below_mass["dust_detected"].mean() if len(below_mass) > 0 else 0

    tp_mass = ((sub["log_Mstar"] >= mass_thresh) & actual_dusty).sum()
    fp_mass = ((sub["log_Mstar"] >= mass_thresh) & (~actual_dusty)).sum()
    fn_mass = ((sub["log_Mstar"] < mass_thresh) & actual_dusty).sum()
    tn_mass = ((sub["log_Mstar"] < mass_thresh) & (~actual_dusty)).sum()
    f1_mass = 2 * tp_mass / (2 * tp_mass + fp_mass + fn_mass) if (2 * tp_mass + fp_mass + fn_mass) > 0 else 0

    result = {
        "n_total": n,
        "dust_detection_threshold": DUST_DETECTION_THRESHOLD,
        "agb_threshold_gyr": AGB_THRESHOLD_GYR,
        "tep_boundary": {
            "n_above_threshold": len(above_teff),
            "n_below_threshold": len(below_teff),
            "dust_frac_above": float(frac_above),
            "dust_frac_below": float(frac_below),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        },
        "z_gt_8": {
            "n": len(z8),
            "n_above": len(above_z8),
            "n_below": len(below_z8),
            "dust_frac_above": float(frac_above_z8),
            "dust_frac_below": float(frac_below_z8),
        },
        "mass_only_comparison": {
            "mass_threshold": float(mass_thresh),
            "dust_frac_above": float(frac_above_mass),
            "dust_frac_below": float(frac_below_mass),
            "f1_mass": float(f1_mass),
            "f1_tep_advantage": float(f1 - f1_mass),
        },
        "boundary_curve": {
            "z": z_boundary.tolist(),
            "log_mstar": [float(m) if not np.isnan(m) else None for m in mstar_boundary],
        },
    }

    print_status(f"  N = {n} galaxies")
    print_status(f"  TEP boundary: F1 = {f1:.3f} (precision={precision:.3f}, recall={recall:.3f})")
    print_status(f"  Mass-only:    F1 = {f1_mass:.3f}")
    print_status(f"  TEP advantage: ΔF1 = {f1 - f1_mass:+.3f}")
    print_status(f"  z>8: dust frac above AGB = {frac_above_z8:.3f} ({len(above_z8)}), below = {frac_below_z8:.3f} ({len(below_z8)})")

    return result, sub, z_boundary, mstar_boundary


# ── Test 2: Activation Curve Fit ─────────────────────────────────────────────

def test2_activation_curve(df):
    """
    Fit the redshift dependence of rho(Gamma_t, dust) to competing forms.
    """
    print_status("TEST 2: Activation Curve Fit")

    z_col = "z" if "z" in df.columns else "z_phot"
    sub = df.dropna(subset=["gamma_t", "dust", z_col]).copy()

    # Fine z-bins
    z_bins = [
        (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0),
        (8.0, 8.5), (8.5, 9.0), (9.0, 10.0),
    ]

    bin_data = []
    for z_lo, z_hi in z_bins:
        mask = (sub[z_col] >= z_lo) & (sub[z_col] < z_hi)
        s = sub[mask]
        if len(s) < 15:
            continue
        rho, p = stats.spearmanr(s["gamma_t"], s["dust"])
        z_mid = (z_lo + z_hi) / 2

        # Bootstrap CI for rho
        rng = np.random.default_rng(42)
        boot_rhos = []
        for _ in range(500):
            idx = rng.choice(len(s), len(s), replace=True)
            r, _ = stats.spearmanr(s["gamma_t"].values[idx], s["dust"].values[idx])
            boot_rhos.append(r)
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        bin_data.append({
            "z_lo": z_lo, "z_hi": z_hi, "z_mid": z_mid,
            "n": len(s), "rho": float(rho), "p": float(p),
            "rho_ci_lo": float(ci_lo), "rho_ci_hi": float(ci_hi),
        })

    if len(bin_data) < 4:
        return {"status": "INSUFFICIENT_BINS"}

    z_mids = np.array([b["z_mid"] for b in bin_data])
    rhos = np.array([b["rho"] for b in bin_data])
    rho_errs = np.array([(b["rho_ci_hi"] - b["rho_ci_lo"]) / (2 * 1.96) for b in bin_data])
    rho_errs = np.maximum(rho_errs, 0.02)  # floor

    # Fit competing models
    models = {}

    # Model 1: TEP — rho(z) = A * (sqrt(1+z) - sqrt(1+z_off))
    def tep_model(z, A, z_off):
        val = np.sqrt(1 + z) - np.sqrt(1 + z_off)
        return A * np.maximum(val, 0)

    try:
        popt, pcov = optimize.curve_fit(
            tep_model, z_mids, rhos, p0=[0.5, 6.0],
            sigma=rho_errs, absolute_sigma=True, maxfev=5000
        )
        rho_pred = tep_model(z_mids, *popt)
        residuals = rhos - rho_pred
        ss_res = np.sum((residuals / rho_errs) ** 2)
        k = 2
        n = len(z_mids)
        aic = ss_res + 2 * k
        bic = ss_res + k * np.log(n)
        models["tep_sqrt"] = {
            "params": {"A": float(popt[0]), "z_off": float(popt[1])},
            "rho_pred": rho_pred.tolist(),
            "chi2": float(ss_res), "k": k, "aic": float(aic), "bic": float(bic),
        }
    except Exception as e:
        models["tep_sqrt"] = {"error": str(e)}

    # Model 2: Linear — rho(z) = a + b*z
    def linear_model(z, a, b):
        return a + b * z

    try:
        popt, _ = optimize.curve_fit(
            linear_model, z_mids, rhos, p0=[-1.0, 0.15],
            sigma=rho_errs, absolute_sigma=True
        )
        rho_pred = linear_model(z_mids, *popt)
        residuals = rhos - rho_pred
        ss_res = np.sum((residuals / rho_errs) ** 2)
        k = 2
        aic = ss_res + 2 * k
        bic = ss_res + k * np.log(n)
        models["linear"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "rho_pred": rho_pred.tolist(),
            "chi2": float(ss_res), "k": k, "aic": float(aic), "bic": float(bic),
        }
    except Exception as e:
        models["linear"] = {"error": str(e)}

    # Model 3: Step function — rho(z) = 0 for z<z_c, rho_max for z>=z_c
    def step_model(z, rho_max, z_c):
        return rho_max * (1 / (1 + np.exp(-5 * (z - z_c))))

    try:
        popt, _ = optimize.curve_fit(
            step_model, z_mids, rhos, p0=[0.6, 7.5],
            sigma=rho_errs, absolute_sigma=True, maxfev=5000
        )
        rho_pred = step_model(z_mids, *popt)
        residuals = rhos - rho_pred
        ss_res = np.sum((residuals / rho_errs) ** 2)
        k = 2
        aic = ss_res + 2 * k
        bic = ss_res + k * np.log(n)
        models["step"] = {
            "params": {"rho_max": float(popt[0]), "z_c": float(popt[1])},
            "rho_pred": rho_pred.tolist(),
            "chi2": float(ss_res), "k": k, "aic": float(aic), "bic": float(bic),
        }
    except Exception as e:
        models["step"] = {"error": str(e)}

    # Model 4: Constant
    rho_mean = np.mean(rhos)
    residuals = rhos - rho_mean
    ss_res = np.sum((residuals / rho_errs) ** 2)
    k = 1
    aic = ss_res + 2 * k
    bic = ss_res + k * np.log(n)
    models["constant"] = {
        "params": {"rho_mean": float(rho_mean)},
        "rho_pred": [float(rho_mean)] * len(z_mids),
        "chi2": float(ss_res), "k": k, "aic": float(aic), "bic": float(bic),
    }

    # Model 5: Quadratic
    def quad_model(z, a, b, c):
        return a + b * z + c * z ** 2

    try:
        popt, _ = optimize.curve_fit(
            quad_model, z_mids, rhos, p0=[0, 0, 0.01],
            sigma=rho_errs, absolute_sigma=True
        )
        rho_pred = quad_model(z_mids, *popt)
        residuals = rhos - rho_pred
        ss_res = np.sum((residuals / rho_errs) ** 2)
        k = 3
        aic = ss_res + 2 * k
        bic = ss_res + k * np.log(n)
        models["quadratic"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])},
            "rho_pred": rho_pred.tolist(),
            "chi2": float(ss_res), "k": k, "aic": float(aic), "bic": float(bic),
        }
    except Exception as e:
        models["quadratic"] = {"error": str(e)}

    # Rank by AIC
    valid_models = {k: v for k, v in models.items() if "aic" in v}
    if valid_models:
        best_aic = min(v["aic"] for v in valid_models.values())
        ranking = sorted(valid_models.items(), key=lambda x: x[1]["aic"])
        for name, m in valid_models.items():
            m["delta_aic"] = float(m["aic"] - best_aic)
        winner = ranking[0][0]
    else:
        winner = "none"
        ranking = []

    result = {
        "z_bins": bin_data,
        "models": models,
        "ranking": [{"model": k, "aic": v["aic"], "delta_aic": v.get("delta_aic", 0)}
                     for k, v in ranking],
        "winner": winner,
        "interpretation": (
            f"Best-fit model: {winner} (ΔAIC = 0). "
            + (f"TEP √(1+z) form {'wins' if winner == 'tep_sqrt' else 'loses'} "
               f"against {len(valid_models)-1} alternatives."
               if "tep_sqrt" in valid_models else "TEP model failed to fit.")
        ),
    }

    for name, m in valid_models.items():
        print_status(f"  {name:12s}: χ²={m['chi2']:.2f}, AIC={m['aic']:.2f}, ΔAIC={m.get('delta_aic',0):+.2f}")
    print_status(f"  Winner: {winner}")

    return result


# ── Test 3: Spectroscopic Sharpening ─────────────────────────────────────────

def test3_spectroscopic_sharpening(df_phot, df_spec):
    """
    Compare TEP signal strength using photo-z vs spec-z.
    """
    print_status("TEST 3: Spectroscopic Sharpening")

    if df_spec is None or len(df_spec) < 20:
        return {"status": "INSUFFICIENT_SPEC_DATA", "n": 0 if df_spec is None else len(df_spec)}

    spec = df_spec.dropna(subset=["gamma_t", "dust", "z"]).copy()
    n_spec = len(spec)

    if n_spec < 20:
        return {"status": "INSUFFICIENT_SPEC_DATA", "n": n_spec}

    # Spec-z correlations
    rho_spec, p_spec = stats.spearmanr(spec["gamma_t"], spec["dust"])
    
    from scripts.utils.p_value_utils import format_p_value
    p_spec_fmt = format_p_value(p_spec)

    # Find matching galaxies in photo-z sample for fair comparison
    # Use the same redshift range as spec sample
    z_min, z_max = spec["z"].min(), spec["z"].max()
    phot = df_phot.dropna(subset=["gamma_t", "dust", "z"]).copy()
    phot_matched = phot[(phot["z"] >= z_min) & (phot["z"] <= z_max)]

    rho_phot_full, p_phot_full = stats.spearmanr(phot_matched["gamma_t"], phot_matched["dust"])
    p_phot_fmt = format_p_value(p_phot_full)

    # Bootstrap CI for spec rho
    rng = np.random.default_rng(42)
    spec_boots = []
    for _ in range(1000):
        idx = rng.choice(n_spec, n_spec, replace=True)
        r, _ = stats.spearmanr(spec["gamma_t"].values[idx], spec["dust"].values[idx])
        spec_boots.append(r)
    ci_lo, ci_hi = np.percentile(spec_boots, [2.5, 97.5])

    # z > 8 subsample
    spec_z8 = spec[spec["z"] >= 8.0]
    phot_z8 = phot_matched[phot_matched["z"] >= 8.0]

    spec_z8_result = {}
    phot_z8_result = {}

    if len(spec_z8) >= 10:
        r, p = stats.spearmanr(spec_z8["gamma_t"], spec_z8["dust"])
        spec_z8_result = {"n": len(spec_z8), "rho": float(r), "p": format_p_value(p)}
    else:
        spec_z8_result = {"n": len(spec_z8), "insufficient": True}

    if len(phot_z8) >= 10:
        r, p = stats.spearmanr(phot_z8["gamma_t"], phot_z8["dust"])
        phot_z8_result = {"n": len(phot_z8), "rho": float(r), "p": format_p_value(p)}

    # Per-survey breakdown if source_catalog exists
    survey_results = {}
    if "source_catalog" in spec.columns:
        for cat in spec["source_catalog"].unique():
            s = spec[spec["source_catalog"] == cat]
            if len(s) >= 10:
                r, p = stats.spearmanr(s["gamma_t"], s["dust"])
                survey_results[cat] = {"n": len(s), "rho": float(r), "p": format_p_value(p)}

    sharpened = rho_spec > rho_phot_full

    result = {
        "n_spec": n_spec,
        "z_range": [float(z_min), float(z_max)],
        "spec_z": {
            "rho": float(rho_spec),
            "p": p_spec_fmt,
            "ci_95": [float(ci_lo), float(ci_hi)],
        },
        "photo_z_matched": {
            "n": len(phot_matched),
            "rho": float(rho_phot_full),
            "p": p_phot_fmt,
        },
        "z_gt_8": {
            "spec": spec_z8_result,
            "phot": phot_z8_result,
        },
        "survey_breakdown": survey_results,
        "sharpened": bool(sharpened),
        "delta_rho": float(rho_spec - rho_phot_full),
        "interpretation": (
            f"Spec-z ρ = {rho_spec:.3f} vs photo-z ρ = {rho_phot_full:.3f} "
            f"(Δρ = {rho_spec - rho_phot_full:+.3f}). "
            + ("Signal SHARPENS with exact redshifts — consistent with photo-z "
               "dilution, ruling out photo-z artifacts as the source."
               if sharpened else
               "Signal does NOT sharpen — photo-z noise is not the dominant factor.")
        ),
    }

    print_status(f"  Spec-z: ρ = {rho_spec:.3f} (N = {n_spec}, p = {p_spec_fmt:.2e})")
    print_status(f"  Photo-z: ρ = {rho_phot_full:.3f} (N = {len(phot_matched)})")
    print_status(f"  Δρ = {rho_spec - rho_phot_full:+.3f} → {'SHARPENED' if sharpened else 'NOT sharpened'}")

    return result


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(test1_result, test1_df, z_boundary, mstar_boundary,
                test2_result, test3_result):
    """Generate 4-panel summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        import sys
        from pathlib import Path
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE
        except ImportError:
            pass
            
    except ImportError:
        return

    try:
        set_pub_style()
    except NameError:
        pass
        
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE['web_quad'])

    # ── Panel 1: AGB Phase Boundary Map ──────────────────────────────────
    ax = axes[0, 0]
    sub = test1_df.copy()

    dust_detected = sub[sub["dust"] > DUST_DETECTION_THRESHOLD]
    dust_free = sub[sub["dust"] <= DUST_DETECTION_THRESHOLD]

    ax.scatter(dust_free["z"], dust_free["log_Mstar"],
               c=COLORS["accent"], alpha=0.3, s=8, label=f"Dust-free (N={len(dust_free)})", zorder=2)
    ax.scatter(dust_detected["z"], dust_detected["log_Mstar"],
               c=COLORS["primary"], alpha=0.3, s=8, label=f"Dusty (N={len(dust_detected)})", zorder=2)

    # TEP boundary
    valid = ~np.isnan(mstar_boundary)
    ax.plot(z_boundary[valid], mstar_boundary[valid],
            "k-", linewidth=2.5, label=r"TEP: $t_{\rm eff} = 0.3$ Gyr", zorder=5)

    # Mass-only threshold
    mass_thresh = test1_result.get("mass_only_comparison", {}).get("mass_threshold", 9.0)
    ax.axhline(mass_thresh, color="gray", linestyle="--", linewidth=1.5,
               label=f"Mass-only: $\\log M_* = {mass_thresh:.1f}$", zorder=4)

    ax.set_xlabel("Redshift $z$", fontsize=12)
    ax.set_ylabel(r"$\log\,M_*/M_\odot$", fontsize=12)
    ax.set_title("AGB Dust Phase Boundary", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(3.5, 10.5)
    ax.set_ylim(6, 11.5)

    # Annotate
    f1_tep = test1_result["tep_boundary"]["f1"]
    f1_mass = test1_result["mass_only_comparison"]["f1_mass"]
    ax.text(0.97, 0.03,
            f"TEP F1={f1_tep:.3f}\nMass F1={f1_mass:.3f}\nΔF1={f1_tep-f1_mass:+.3f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ── Panel 2: Activation Curve ────────────────────────────────────────
    ax = axes[0, 1]
    bins = test2_result.get("z_bins", [])
    if bins:
        z_mids = [b["z_mid"] for b in bins]
        rhos = [b["rho"] for b in bins]
        ci_lo = [b["rho_ci_lo"] for b in bins]
        ci_hi = [b["rho_ci_hi"] for b in bins]
        yerr_lo = [r - lo for r, lo in zip(rhos, ci_lo)]
        yerr_hi = [hi - r for r, hi in zip(rhos, ci_hi)]

        ax.errorbar(z_mids, rhos, yerr=[yerr_lo, yerr_hi],
                     fmt="ko", markersize=8, capsize=4, capthick=1.5,
                     elinewidth=1.5, zorder=5, label="Data")

        # Overlay model fits
        z_fine = np.linspace(3.5, 10.5, 200)
        models = test2_result.get("models", {})
        colors = {"tep_sqrt": COLORS["primary"], "linear": COLORS["accent"],
                  "step": COLORS["secondary"], "quadratic": COLORS["highlight"], "constant": COLORS["gray"]}
        labels = {"tep_sqrt": r"TEP: $\sqrt{1+z}$", "linear": "Linear",
                  "step": "Step function", "quadratic": "Quadratic", "constant": "Constant"}

        for name in ["tep_sqrt", "linear", "step", "quadratic", "constant"]:
            m = models.get(name, {})
            params = m.get("params", {})
            if not params:
                continue
            if name == "tep_sqrt" and "A" in params:
                y = params["A"] * np.maximum(np.sqrt(1 + z_fine) - np.sqrt(1 + params["z_off"]), 0)
            elif name == "linear" and "a" in params:
                y = params["a"] + params["b"] * z_fine
            elif name == "step" and "rho_max" in params:
                y = params["rho_max"] / (1 + np.exp(-5 * (z_fine - params["z_c"])))
            elif name == "quadratic" and "a" in params:
                y = params["a"] + params["b"] * z_fine + params["c"] * z_fine ** 2
            elif name == "constant":
                y = np.full_like(z_fine, params["rho_mean"])
            else:
                continue

            delta = m.get("delta_aic", 99)
            lbl = f"{labels[name]} (ΔAIC={delta:+.1f})"
            lw = 2.5 if delta == 0 else 1.2
            ls = "-" if delta == 0 else "--"
            ax.plot(z_fine, y, color=colors[name], linewidth=lw, linestyle=ls, label=lbl)

        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.set_xlabel("Redshift $z$", fontsize=12)
        ax.set_ylabel(r"$\rho(\Gamma_t,\,\mathrm{dust})$", fontsize=12)
        ax.set_title("Activation Curve: Which Form Fits?", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlim(3.5, 10.5)
        ax.set_ylim(-0.3, 0.9)

    # ── Panel 3: Spectroscopic sharpening ────────────────────────────────
    ax = axes[1, 0]
    t3 = test3_result
    if "spec_z" in t3 and "photo_z_matched" in t3:
        labels_bar = ["Photo-$z$\n(matched)", "Spec-$z$"]
        rhos_bar = [t3["photo_z_matched"]["rho"], t3["spec_z"]["rho"]]
        ns = [t3["photo_z_matched"]["n"], t3["n_spec"]]
        colors_bar = [COLORS["gray"], COLORS["primary"]]
        bars = ax.bar(labels_bar, rhos_bar, color=colors_bar, edgecolor="black", linewidth=0.8)

        # CI on spec
        ci = t3["spec_z"].get("ci_95", [0, 0])
        ax.errorbar(1, rhos_bar[1], yerr=[[rhos_bar[1] - ci[0]], [ci[1] - rhos_bar[1]]],
                     fmt="none", color="black", capsize=6, capthick=2, elinewidth=2)

        for bar, r, n in zip(bars, rhos_bar, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, max(r, 0) + 0.02,
                    f"ρ={r:.3f}\nN={n}", ha="center", va="bottom", fontsize=10)

        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.set_ylabel(r"$\rho(\Gamma_t,\,\mathrm{dust})$", fontsize=12)
        delta = t3.get("delta_rho", 0)
        ax.set_title(f"Spec-z Sharpening (Δρ = {delta:+.3f})", fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Insufficient spec-z data", transform=ax.transAxes,
                ha="center", fontsize=14)
        ax.set_title("Spec-z Sharpening", fontsize=13, fontweight="bold")

    # ── Panel 4: Summary ─────────────────────────────────────────────────
    ax = axes[1, 1]

    text = "PHASE BOUNDARY + ACTIVATION RESULTS\n"
    text += "=" * 38 + "\n\n"

    # Phase boundary
    pb = test1_result
    text += f"1. AGB Phase Boundary (t_eff=0.3 Gyr)\n"
    text += f"   TEP F1 = {pb['tep_boundary']['f1']:.3f}\n"
    text += f"   Mass F1 = {pb['mass_only_comparison']['f1_mass']:.3f}\n"
    text += f"   z>8: dusty above AGB = {pb['z_gt_8']['dust_frac_above']:.0%}\n"
    text += f"        dusty below AGB = {pb['z_gt_8']['dust_frac_below']:.0%}\n\n"

    # Activation
    text += f"2. Activation Curve Winner:\n"
    winner = test2_result.get("winner", "?")
    text += f"   {winner}\n"
    ranking = test2_result.get("ranking", [])
    for r in ranking[:4]:
        text += f"   {r['model']:12s} ΔAIC={r['delta_aic']:+.1f}\n"
    text += "\n"

    # Spec sharpening
    if "delta_rho" in t3:
        text += f"3. Spec-z Sharpening: Δρ={t3['delta_rho']:+.3f}\n"
        text += f"   {'SHARPENED' if t3.get('sharpened') else 'NOT sharpened'}\n"

    ax.text(0.05, 0.95, text, ha="left", va="top", fontsize=9.5,
            transform=ax.transAxes, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3))
    ax.set_title("Summary", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.suptitle("Step 145: AGB Phase Boundary, Activation Curve & Spec-z Sharpening",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 145: Phase Boundary + Activation Curve + Spec-z Sharpening")
    logger.info("=" * 70)

    df = load_uncover()
    if df is None:
        logger.error("UNCOVER data not found")
        return

    df_spec = load_specz()
    logger.info(f"UNCOVER: {len(df)} galaxies")
    if df_spec is not None:
        logger.info(f"Spec-z: {len(df_spec)} galaxies")

    results = {
        "step": "Step 145: AGB Phase Boundary + Activation Curve + Spec-z Sharpening",
    }

    # Test 1
    test1_result, test1_df, z_boundary, mstar_boundary = test1_phase_boundary(df)
    results["test_1_phase_boundary"] = test1_result

    # Test 2
    test2_result = test2_activation_curve(df)
    results["test_2_activation_curve"] = test2_result

    # Test 3
    test3_result = test3_spectroscopic_sharpening(df, df_spec)
    results["test_3_spec_sharpening"] = test3_result

    # Overall
    verdicts = []
    f1_adv = test1_result["mass_only_comparison"]["f1_tep_advantage"]
    verdicts.append(f"PHASE BOUNDARY: TEP ΔF1={f1_adv:+.3f} vs mass-only")

    winner = test2_result.get("winner", "?")
    tep_delta = test2_result.get("models", {}).get("tep_sqrt", {}).get("delta_aic", None)
    verdicts.append(
        f"ACTIVATION: {winner} wins by AIC"
        + (f" (TEP ΔAIC={tep_delta:+.1f})" if tep_delta is not None else "")
    )

    if "delta_rho" in test3_result:
        verdicts.append(
            f"SPEC-Z: {'SHARPENED' if test3_result['sharpened'] else 'NOT sharpened'} "
            f"(Δρ={test3_result['delta_rho']:+.3f})"
        )

    results["verdicts"] = verdicts

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"Results saved to {out_path}")

    # Figure
    make_figure(test1_result, test1_df, z_boundary, mstar_boundary,
                test2_result, test3_result)

    print_status("=" * 60)
    print_status("RESULTS")
    print_status("=" * 60)
    for v in verdicts:
        print_status(f"  {v}")

    return results


if __name__ == "__main__":
    main()

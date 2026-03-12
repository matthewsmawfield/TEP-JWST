#!/usr/bin/env python3
"""
Step 148: Mass-Independent Potential-Depth Proxy Tests

The central critique is that Gamma_t is deterministic in M* and z, so any
mass-dependent effect could mimic TEP. To break this degeneracy, we test
whether the TEP signal persists when using proxies for potential depth
that are INDEPENDENT of stellar mass:

Test 1: SFR-Surface-Density Proxy
    Surface SFR density (SFR / angular_size^2) traces gas density and
    compactness — properties correlated with potential depth but partially
    independent of M*. If dust correlates with SFR surface density at
    fixed M* and z, it suggests a potential-depth effect beyond mass.

Test 2: sSFR-Residual as Environment Proxy  
    After removing the mass-sSFR relation, the residual sSFR traces
    local environment (gas accretion, merger history) independently of M*.
    If this residual predicts dust at z>8, it indicates environment-dependent
    dust production beyond mass scaling.

Test 3: Chi-squared Residual as SED-Tension Proxy
    The SED fitting chi-squared measures how well standard templates fit.
    Under TEP, high-Gamma_t galaxies have distorted SEDs that are harder
    to fit, producing higher chi2. If chi2 predicts dust independently of
    M*, it supports the isochrony-bias interpretation.

Test 4: Photometric Redshift Uncertainty as Potential-Depth Proxy
    Under TEP, galaxies in deep potentials have systematically different
    SEDs. The photo-z uncertainty (z_84 - z_16) should correlate with
    Gamma_t because TEP-affected SEDs are harder to fit. This is mass-
    independent because photo-z uncertainty depends on SED shape, not mass.

Test 5: Age-Ratio Residual Independence Test
    After removing mass dependence, the age residual should correlate with
    dust if the age-dust connection is mediated by effective time rather
    than mass alone.

Outputs:
- results/outputs/step_148_mass_independent_proxy.json
- results/figures/figure_148_mass_independent_proxy.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import safe_json_default

STEP_NUM = "148"
STEP_NAME = "mass_independent_proxy"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

for p in [LOGS_PATH, OUTPUT_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def rankdata_with_nan(x):
    """Rank finite values, preserving NaNs."""
    x = np.asarray(x, dtype=float)
    ranks = np.full(x.shape, np.nan, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() > 0:
        ranks[mask] = stats.rankdata(x[mask])
    return ranks


def residualize(x, covariates):
    """Remove linear dependence of x on covariates via OLS residuals."""
    from sklearn.linear_model import LinearRegression
    mask = np.isfinite(x) & np.all(np.isfinite(covariates), axis=1)
    if mask.sum() < 10:
        return x, mask
    reg = LinearRegression().fit(covariates[mask], x[mask])
    resid = np.full_like(x, np.nan)
    resid[mask] = x[mask] - reg.predict(covariates[mask])
    return resid, mask


def partial_spearman(x, y, covariates):
    """Compute rank-based partial correlation controlling for covariates."""
    x_rank = rankdata_with_nan(x)
    y_rank = rankdata_with_nan(y)
    cov_rank = np.column_stack([rankdata_with_nan(covariates[:, i]) for i in range(covariates.shape[1])])
    x_resid, mask_x = residualize(x_rank, cov_rank)
    y_resid, mask_y = residualize(y_rank, cov_rank)
    mask = mask_x & mask_y & np.isfinite(x_resid) & np.isfinite(y_resid)
    if mask.sum() < 10:
        return 0.0, 1.0, 0
    rho, p = stats.pearsonr(x_resid[mask], y_resid[mask])
    return float(rho), float(p), int(mask.sum())


def bootstrap_ci(x, y, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Spearman rho."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return 0.0, -1.0, 1.0
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)
    rhos = sorted(rhos)
    lo = rhos[int(alpha / 2 * n_boot)]
    hi = rhos[int((1 - alpha / 2) * n_boot)]
    return float(np.median(rhos)), float(lo), float(hi)


def main():
    logger.info("=" * 70)
    logger.info("Step 148: Mass-Independent Potential-Depth Proxy Tests")
    logger.info("=" * 70)

    # Load data
    uncover_path = PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv"
    
    if not uncover_path.exists():
        logger.error(f"Input file not found: {uncover_path}")
        return

    df = pd.read_csv(uncover_path)
    print_status(f"Loaded UNCOVER: N = {len(df)}")

    # Prepare columns
    df["z"] = df["z_phot"].fillna(df["z_spec"])
    df["log_mstar"] = df["log_Mstar"]
    df["av"] = df["dust"]
    df["log_sfr"] = np.log10(df["sfr100"].clip(lower=1e-5))
    df["log_ssfr_col"] = df["log_ssfr"]
    df["z_unc"] = df["z_84"] - df["z_16"]
    
    # Create SFR surface density proxy (SFR / z^2 as angular size proxy)
    # At fixed z, luminosity distance is constant, so angular size ~ physical size
    # SFR surface density ~ SFR / (physical_size)^2
    # Without direct size measurements, we use SFR/(M*)^(2/3) as a proxy
    # (virial radius ~ M^(1/3), so surface area ~ M^(2/3))
    df["sfr_surface_density"] = df["log_sfr"] - (2.0 / 3.0) * df["log_mstar"]

    # Chi2 as SED tension proxy
    df["log_chi2"] = np.log10(df["chi2"].clip(lower=0.01))

    results = {
        "step": "Step 148: Mass-Independent Potential-Depth Proxy Tests",
        "description": (
            "Five tests of whether the TEP dust signal persists when using "
            "proxies for potential depth that are partially independent of "
            "stellar mass. Each test computes partial correlations controlling "
            "for M* and z."
        ),
        "N_total": len(df),
        "tests": {},
    }

    # ── Define subsamples ───────────────────────────────────────────────
    z8_mask = df["z"] > 8
    full_mask = (df["z"] > 4) & (df["z"] < 10)
    
    subsamples = {
        "full_z4_10": full_mask,
        "z_gt_8": z8_mask,
    }

    for sample_name, sample_mask in subsamples.items():
        dfs = df[sample_mask].copy().reset_index(drop=True)
        N = len(dfs)
        print_status(f"\n{'='*50}")
        print_status(f"Sample: {sample_name} (N = {N})")
        print_status(f"{'='*50}")

        if N < 30:
            results["tests"][sample_name] = {"N": N, "skipped": "too few galaxies"}
            continue

        sample_results = {"N": N}
        covs = dfs[["log_mstar", "z"]].values

        # Recompute residual proxies within each subsample to avoid importing
        # lower-z trend structure into the z>8 analysis.
        dfs["ssfr_resid"], _ = residualize(
            rankdata_with_nan(dfs["log_ssfr_col"].values),
            np.column_stack([
                rankdata_with_nan(dfs["log_mstar"].values),
                rankdata_with_nan(dfs["z"].values),
            ]),
        )
        dfs["age_ratio_resid"], _ = residualize(
            rankdata_with_nan(dfs["age_ratio"].values),
            np.column_stack([
                rankdata_with_nan(dfs["log_mstar"].values),
                rankdata_with_nan(dfs["z"].values),
            ]),
        )

        # ── Test 1: SFR Surface Density ─────────────────────────────────
        print_status("Test 1: SFR Surface Density Proxy...")
        
        # Raw correlation
        mask_t1 = np.isfinite(dfs["sfr_surface_density"]) & np.isfinite(dfs["av"])
        if mask_t1.sum() > 10:
            rho_raw, p_raw = stats.spearmanr(
                dfs.loc[mask_t1, "sfr_surface_density"],
                dfs.loc[mask_t1, "av"],
            )
        else:
            rho_raw, p_raw = 0.0, 1.0

        # Partial (controlling for M* and z)
        rho_part, p_part, n_part = partial_spearman(
            dfs["sfr_surface_density"].values, dfs["av"].values, covs
        )

        rho_proxy_gt, p_proxy_gt, n_proxy_gt = partial_spearman(
            dfs["sfr_surface_density"].values, dfs["gamma_t"].values, covs
        )

        # Also test: does SFR surface density predict dust AFTER controlling for Gamma_t?
        covs_gt = dfs[["log_mstar", "z", "gamma_t"]].values
        rho_after_gt, p_after_gt, n_after_gt = partial_spearman(
            dfs["sfr_surface_density"].values, dfs["av"].values, covs_gt
        )

        sample_results["test1_sfr_surface_density"] = {
            "description": "SFR/(M*^(2/3)) as potential-depth proxy",
            "rho_raw": float(rho_raw),
            "p_raw": float(p_raw),
            "rho_partial_Mz": float(rho_part),
            "p_partial_Mz": float(p_part),
            "N_partial": n_part,
            "rho_surface_density_gamma_t_partial": float(rho_proxy_gt),
            "p_surface_density_gamma_t_partial": float(p_proxy_gt),
            "N_partial_gamma_t": n_proxy_gt,
            "rho_partial_Mz_Gt": float(rho_after_gt),
            "p_partial_Mz_Gt": float(p_after_gt),
            "N_partial_Gt": n_after_gt,
            "mass_independent_signal": bool(p_part < 0.05),
            "dust_significant": bool(p_part < 0.05),
            "gamma_linked_signal": bool(p_proxy_gt < 0.05),
            "joint_supportive_signal": bool(
                (p_part < 0.05) and (p_proxy_gt < 0.05) and (rho_part * rho_proxy_gt > 0)
            ),
            "survives_gamma_t_control": bool(p_after_gt < 0.05),
        }
        print_status(f"  Raw: ρ={rho_raw:.3f}, p={p_raw:.2e}")
        print_status(f"  Partial (M*,z): ρ={rho_part:.3f}, p={p_part:.2e}")
        print_status(f"  Proxy→Γt (partial M*,z): ρ={rho_proxy_gt:.3f}, p={p_proxy_gt:.2e}")
        print_status(f"  After Γt control: ρ={rho_after_gt:.3f}, p={p_after_gt:.2e}")

        # ── Test 2: sSFR Residual ───────────────────────────────────────
        print_status("Test 2: sSFR Residual Proxy...")
        
        ssfr_resid_vals = dfs["ssfr_resid"].values
        mask_t2 = np.isfinite(ssfr_resid_vals) & np.isfinite(dfs["av"].values)
        
        if mask_t2.sum() > 10:
            rho_ssfr, p_ssfr = stats.spearmanr(
                ssfr_resid_vals[mask_t2], dfs["av"].values[mask_t2]
            )
        else:
            rho_ssfr, p_ssfr = 0.0, 1.0
            
        # sSFR residual vs Gamma_t (should correlate if TEP is real)
        mask_t2b = mask_t2 & np.isfinite(dfs["gamma_t"].values)
        if mask_t2b.sum() > 10:
            rho_ssfr_gt, p_ssfr_gt = stats.spearmanr(
                ssfr_resid_vals[mask_t2b], dfs["gamma_t"].values[mask_t2b]
            )
        else:
            rho_ssfr_gt, p_ssfr_gt = 0.0, 1.0

        sample_results["test2_ssfr_residual"] = {
            "description": "sSFR residual after removing mass+z trend → dust",
            "rho_ssfr_resid_dust": float(rho_ssfr),
            "p_ssfr_resid_dust": float(p_ssfr),
            "N": int(mask_t2.sum()),
            "rho_ssfr_resid_gamma_t": float(rho_ssfr_gt),
            "p_ssfr_resid_gamma_t": float(p_ssfr_gt),
            "dust_significant": bool(p_ssfr < 0.05),
            "gamma_linked_signal": bool(p_ssfr_gt < 0.05),
            "joint_supportive_signal": bool(
                (p_ssfr < 0.05) and (p_ssfr_gt < 0.05) and (rho_ssfr * rho_ssfr_gt > 0)
            ),
            "interpretation": (
                "If sSFR residual (mass-independent) predicts dust, this supports "
                "an effect beyond mass scaling."
            ),
        }
        print_status(f"  sSFR_resid → dust: ρ={rho_ssfr:.3f}, p={p_ssfr:.2e}")
        print_status(f"  sSFR_resid → Γt: ρ={rho_ssfr_gt:.3f}, p={p_ssfr_gt:.2e}")

        # ── Test 3: Chi-squared as SED Tension Proxy ────────────────────
        print_status("Test 3: χ² SED Tension Proxy...")
        
        rho_chi2, p_chi2, n_chi2 = partial_spearman(
            dfs["log_chi2"].values, dfs["av"].values, covs
        )
        
        # chi2 vs gamma_t (partial)
        rho_chi2_gt, p_chi2_gt, n_chi2_gt = partial_spearman(
            dfs["log_chi2"].values, dfs["gamma_t"].values, covs
        )

        sample_results["test3_chi2_proxy"] = {
            "description": "SED chi² as TEP distortion proxy → dust",
            "rho_chi2_dust_partial": float(rho_chi2),
            "p_chi2_dust_partial": float(p_chi2),
            "N": n_chi2,
            "rho_chi2_gamma_t_partial": float(rho_chi2_gt),
            "p_chi2_gamma_t_partial": float(p_chi2_gt),
            "dust_significant": bool(p_chi2 < 0.05),
            "gamma_linked_signal": bool(p_chi2_gt < 0.05),
            "joint_supportive_signal": bool(
                (p_chi2 < 0.05) and (p_chi2_gt < 0.05) and (rho_chi2 * rho_chi2_gt > 0)
            ),
            "interpretation": (
                "Higher chi² indicates poorer SED fit. Under TEP, high-Gamma_t "
                "galaxies have distorted SEDs. If chi² predicts dust at fixed M* "
                "and z, it supports the isochrony-bias interpretation."
            ),
        }
        print_status(f"  χ²→dust (partial M*,z): ρ={rho_chi2:.3f}, p={p_chi2:.2e}")
        print_status(f"  χ²→Γt (partial M*,z): ρ={rho_chi2_gt:.3f}, p={p_chi2_gt:.2e}")

        # ── Test 4: Photo-z Uncertainty Proxy ───────────────────────────
        print_status("Test 4: Photo-z Uncertainty Proxy...")
        
        rho_zunc, p_zunc, n_zunc = partial_spearman(
            dfs["z_unc"].values, dfs["av"].values, covs
        )
        
        rho_zunc_gt, p_zunc_gt, n_zunc_gt = partial_spearman(
            dfs["z_unc"].values, dfs["gamma_t"].values, covs
        )

        sample_results["test4_photoz_uncertainty"] = {
            "description": "Photo-z uncertainty (z_84-z_16) as SED-shape proxy → dust",
            "rho_zunc_dust_partial": float(rho_zunc),
            "p_zunc_dust_partial": float(p_zunc),
            "N": n_zunc,
            "rho_zunc_gamma_t_partial": float(rho_zunc_gt),
            "p_zunc_gamma_t_partial": float(p_zunc_gt),
            "dust_significant": bool(p_zunc < 0.05),
            "gamma_linked_signal": bool(p_zunc_gt < 0.05),
            "joint_supportive_signal": bool(
                (p_zunc < 0.05) and (p_zunc_gt < 0.05) and (rho_zunc * rho_zunc_gt > 0)
            ),
            "interpretation": (
                "Photo-z uncertainty depends on SED shape, not mass directly. "
                "If it predicts dust at fixed M* and z, this is a mass-independent "
                "signal consistent with TEP-distorted SEDs."
            ),
        }
        print_status(f"  Δz→dust (partial M*,z): ρ={rho_zunc:.3f}, p={p_zunc:.2e}")
        print_status(f"  Δz→Γt (partial M*,z): ρ={rho_zunc_gt:.3f}, p={p_zunc_gt:.2e}")

        # ── Test 5: Age-Ratio Residual ──────────────────────────────────
        print_status("Test 5: Age-Ratio Residual...")
        
        age_resid_vals = dfs["age_ratio_resid"].values
        mask_t5 = np.isfinite(age_resid_vals) & np.isfinite(dfs["av"].values)
        
        if mask_t5.sum() > 10:
            rho_age, p_age = stats.spearmanr(
                age_resid_vals[mask_t5], dfs["av"].values[mask_t5]
            )
        else:
            rho_age, p_age = 0.0, 1.0

        mask_t5_gt = mask_t5 & np.isfinite(dfs["gamma_t"].values)
        if mask_t5_gt.sum() > 10:
            rho_age_gt, p_age_gt = stats.spearmanr(
                age_resid_vals[mask_t5_gt], dfs["gamma_t"].values[mask_t5_gt]
            )
        else:
            rho_age_gt, p_age_gt = 0.0, 1.0

        sample_results["test5_age_residual"] = {
            "description": "Age-ratio residual (after M*+z control) → dust",
            "rho_age_resid_dust": float(rho_age),
            "p_age_resid_dust": float(p_age),
            "N": int(mask_t5.sum()),
            "rho_age_resid_gamma_t": float(rho_age_gt),
            "p_age_resid_gamma_t": float(p_age_gt),
            "dust_significant": bool(p_age < 0.05),
            "gamma_linked_signal": bool(p_age_gt < 0.05),
            "joint_supportive_signal": bool(
                (p_age < 0.05) and (p_age_gt < 0.05) and (rho_age * rho_age_gt > 0)
            ),
            "interpretation": (
                "If the mass-independent component of age ratio predicts dust, "
                "this supports a time-dependent mechanism (TEP) rather than "
                "a pure mass proxy."
            ),
        }
        print_status(f"  age_resid → dust: ρ={rho_age:.3f}, p={p_age:.2e}")
        print_status(f"  age_resid → Γt: ρ={rho_age_gt:.3f}, p={p_age_gt:.2e}")

        # ── Combined Assessment ─────────────────────────────────────────
        proxy_tests = [
            sample_results["test1_sfr_surface_density"],
            sample_results["test2_ssfr_residual"],
            sample_results["test3_chi2_proxy"],
            sample_results["test4_photoz_uncertainty"],
            sample_results["test5_age_residual"],
        ]
        n_dust_significant = sum(t.get("dust_significant", False) for t in proxy_tests)
        n_gamma_linked = sum(t.get("gamma_linked_signal", False) for t in proxy_tests)
        n_joint_supportive = sum(t.get("joint_supportive_signal", False) for t in proxy_tests)
        
        sample_results["summary"] = {
            "tests_significant_at_005": int(n_dust_significant),
            "tests_with_dust_association": int(n_dust_significant),
            "tests_with_gamma_link": int(n_gamma_linked),
            "tests_jointly_supportive": int(n_joint_supportive),
            "tests_total": 5,
            "fraction_significant": n_dust_significant / 5,
            "assessment": (
                f"{n_dust_significant}/5 indirect proxies significantly predict dust "
                f"at fixed M* and z in the {sample_name} sample; "
                f"{n_gamma_linked}/5 also track Γt directly; "
                f"{n_joint_supportive}/5 satisfy both conditions with a consistent depth ordering."
            ),
        }

        results["tests"][sample_name] = sample_results

    # ── Overall Summary ─────────────────────────────────────────────────
    z8_tests = results["tests"].get("z_gt_8", {})
    z8_summary = z8_tests.get("summary", {})
    
    results["headline"] = {
        "z_gt_8_tests_passing": z8_summary.get("tests_jointly_supportive", 0),
        "z_gt_8_tests_total": 5,
        "z_gt_8_tests_with_dust_association": z8_summary.get("tests_with_dust_association", 0),
        "z_gt_8_tests_with_gamma_link": z8_summary.get("tests_with_gamma_link", 0),
        "conclusion": (
            f"At z>8, {z8_summary.get('tests_with_dust_association', 0)}/5 indirect proxies "
            f"associate with dust after mass+z control, but only {z8_summary.get('tests_with_gamma_link', 0)}/5 "
            f"also track Γt directly and {z8_summary.get('tests_jointly_supportive', 0)}/5 satisfy both criteria "
            f"with a consistent depth ordering. The proxy suite therefore narrows — but does not close — the "
            f"mass-proxy degeneracy."
        ),
        "honest_caveat": (
            "All five proxies are INDIRECT measures of potential depth. "
            "They are partially correlated with mass through astrophysical "
            "channels unrelated to TEP. A definitive test requires a "
            "mass-independent dynamical measurement of potential depth, "
            "such as NIRSpec IFU velocity dispersions for z>8 galaxies. "
            "This test is proposed as a Cycle 4 JWST program."
        ),
    }

    # Save
    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    logger.info(f"Results saved to {out_path}")

    # Figure
    make_figure(results, df, z8_mask)

    print_status("=" * 60)
    print_status("MASS-INDEPENDENT PROXY TEST RESULTS")
    print_status("=" * 60)
    for sample_name, sample_data in results["tests"].items():
        if "summary" in sample_data:
            s = sample_data["summary"]
            print_status(
                f"  {sample_name}: {s['tests_with_dust_association']}/5 dust-associated proxies; "
                f"{s['tests_jointly_supportive']}/5 jointly Γt-linked"
            )

    return results


def make_figure(results, df, z8_mask):
    """Generate summary figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    dfs = df[z8_mask].copy()
    N = len(dfs)
    cov_rank = np.column_stack([
        rankdata_with_nan(dfs["log_mstar"].values),
        rankdata_with_nan(dfs["z"].values),
    ])
    dfs["ssfr_resid"], _ = residualize(rankdata_with_nan(dfs["log_ssfr_col"].values), cov_rank)
    dfs["age_ratio_resid"], _ = residualize(rankdata_with_nan(dfs["age_ratio"].values), cov_rank)

    # Panel 1: SFR surface density vs dust
    ax = axes[0, 0]
    mask = np.isfinite(dfs["sfr_surface_density"]) & np.isfinite(dfs["av"])
    if mask.sum() > 0:
        sc = ax.scatter(
            dfs.loc[mask, "sfr_surface_density"],
            dfs.loc[mask, "av"],
            c=dfs.loc[mask, "gamma_t"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\Gamma_t$")
    t1 = results["tests"].get("z_gt_8", {}).get("test1_sfr_surface_density", {})
    ax.set_xlabel("SFR Surface Density Proxy")
    ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title(f"Test 1: SFR Surface Density\npartial ρ={t1.get('rho_partial_Mz', 0):.3f}")

    # Panel 2: sSFR residual vs dust
    ax = axes[0, 1]
    mask = np.isfinite(dfs["ssfr_resid"]) & np.isfinite(dfs["av"])
    if mask.sum() > 0:
        sc = ax.scatter(
            dfs.loc[mask, "ssfr_resid"],
            dfs.loc[mask, "av"],
            c=dfs.loc[mask, "gamma_t"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\Gamma_t$")
    t2 = results["tests"].get("z_gt_8", {}).get("test2_ssfr_residual", {})
    ax.set_xlabel("sSFR Residual (mass-controlled)")
    ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title(f"Test 2: sSFR Residual\nρ={t2.get('rho_ssfr_resid_dust', 0):.3f}")

    # Panel 3: chi2 vs dust (partial)
    ax = axes[0, 2]
    mask = np.isfinite(dfs["log_chi2"]) & np.isfinite(dfs["av"])
    if mask.sum() > 0:
        sc = ax.scatter(
            dfs.loc[mask, "log_chi2"],
            dfs.loc[mask, "av"],
            c=dfs.loc[mask, "gamma_t"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\Gamma_t$")
    t3 = results["tests"].get("z_gt_8", {}).get("test3_chi2_proxy", {})
    ax.set_xlabel(r"log($\chi^2$)")
    ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title(f"Test 3: SED χ² Tension\npartial ρ={t3.get('rho_chi2_dust_partial', 0):.3f}")

    # Panel 4: photo-z uncertainty vs dust
    ax = axes[1, 0]
    mask = np.isfinite(dfs["z_unc"]) & np.isfinite(dfs["av"])
    if mask.sum() > 0:
        sc = ax.scatter(
            dfs.loc[mask, "z_unc"],
            dfs.loc[mask, "av"],
            c=dfs.loc[mask, "gamma_t"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\Gamma_t$")
    t4 = results["tests"].get("z_gt_8", {}).get("test4_photoz_uncertainty", {})
    ax.set_xlabel(r"$\Delta z$ ($z_{84} - z_{16}$)")
    ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title(f"Test 4: Photo-z Uncertainty\npartial ρ={t4.get('rho_zunc_dust_partial', 0):.3f}")

    # Panel 5: age residual vs dust
    ax = axes[1, 1]
    mask = np.isfinite(dfs["age_ratio_resid"]) & np.isfinite(dfs["av"])
    if mask.sum() > 0:
        sc = ax.scatter(
            dfs.loc[mask, "age_ratio_resid"],
            dfs.loc[mask, "av"],
            c=dfs.loc[mask, "gamma_t"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\Gamma_t$")
    t5 = results["tests"].get("z_gt_8", {}).get("test5_age_residual", {})
    ax.set_xlabel("Age-Ratio Residual (mass-controlled)")
    ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title(f"Test 5: Age Residual\nρ={t5.get('rho_age_resid_dust', 0):.3f}")

    # Panel 6: Summary
    ax = axes[1, 2]
    z8_data = results["tests"].get("z_gt_8", {})
    summary_lines = ["MASS-INDEPENDENT PROXY TESTS", f"Sample: z > 8 (N = {z8_data.get('N', 0)})", ""]
    
    test_names = [
        ("test1_sfr_surface_density", "SFR Surface Density", "rho_partial_Mz", "p_partial_Mz"),
        ("test2_ssfr_residual", "sSFR Residual", "rho_ssfr_resid_dust", "p_ssfr_resid_dust"),
        ("test3_chi2_proxy", "SED χ² Tension", "rho_chi2_dust_partial", "p_chi2_dust_partial"),
        ("test4_photoz_uncertainty", "Photo-z Uncertainty", "rho_zunc_dust_partial", "p_zunc_dust_partial"),
        ("test5_age_residual", "Age Residual", "rho_age_resid_dust", "p_age_resid_dust"),
    ]
    
    for key, name, rho_key, p_key in test_names:
        t = z8_data.get(key, {})
        rho = t.get(rho_key, 0)
        p = t.get(p_key, 1)
        status = "✓" if p < 0.05 else "✗"
        summary_lines.append(f"{status} {name}: ρ={rho:+.3f} (p={p:.2e})")
    
    s = z8_data.get("summary", {})
    summary_lines.append(
        f"\n{s.get('tests_with_dust_association', 0)}/5 dust-associated; "
        f"{s.get('tests_with_gamma_link', 0)}/5 Γt-linked; "
        f"{s.get('tests_jointly_supportive', 0)}/5 joint"
    )
    
    ax.text(
        0.05, 0.95, "\n".join(summary_lines),
        ha="left", va="top", fontsize=9,
        transform=ax.transAxes, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
    )
    ax.set_title("Summary")
    ax.axis("off")

    plt.suptitle(
        "Step 148: Mass-Independent Potential-Depth Proxy Tests (z > 8)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    fig_path = FIGURES_PATH / f"figure_{STEP_NUM}_{STEP_NAME}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()

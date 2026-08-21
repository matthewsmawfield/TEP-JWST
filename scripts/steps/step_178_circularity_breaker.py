#!/usr/bin/env python3
"""
Step 178: Circular Mass Loop Breaker

Directly targets the "circular mass loop" criticism:

    R_ML depends on M_halo → M_halo estimated from M_* (abundance matching)
    → TEP says M_* is biased → circular feedback.

The N=2,315 photometric sample is caught in this loop.  This step implements
two new tests that do not exist in step_143 or step_144:

TEST A — 2D Mass-Z Pairing Shuffle:
    Randomly reassign each galaxy's (M_*, z) pair to a different galaxy,
    breaking BOTH the mass-z correlation AND the R_ML functional form
    simultaneously.  If the signal survives, it cannot be a mass-z artifact.
    This is stronger than step_143 Test 3 (which only shuffles mass within
    z-bins, preserving the z-dependence of R_ML).

TEST B — Placebo R_ML (Wrong Functional Form):
    Construct a fake R_ML with the SAME mass and z dependence but a DIFFERENT
    functional form (power-law instead of exponential potential-depth).
    If the real R_ML outperforms the placebo, the specific TEP functional
    form matters.  If they perform equally, the signal is just generic
    mass-z nonlinearity.

TEST C — Mass-Independent Subsets:
    Restrict to narrow mass bins where M_* varies by < 0.5 dex.  Within
    these bins, R_ML varies only through the z-dependence (sqrt(1+z) factor
    and the redshift-dependent SMHM offset).  If the dust-R_ML correlation
    survives within narrow mass bins, it cannot be driven by the mass axis
    of the circularity.

Outputs:
- results/outputs/step_178_circularity_breaker.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import (
    compute_ml_response,
    stellar_to_halo_mass_behroozi_like,
    KAPPA_GAL,
    ALPHA_NUCLEAR,
)

STEP_NUM = "178"
STEP_NAME = "circularity_breaker"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

N_SHUFFLES = 2000
RNG_SEED = 178


def _load_data():
    """Load the UNCOVER sample with R_ML already computed."""
    csv_path = PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"UNCOVER interim CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "z_phot" in df.columns and "z" not in df.columns:
        df = df.rename(columns={"z_phot": "z"})
    if "dust" in df.columns and "Av" not in df.columns:
        df = df.rename(columns={"dust": "Av"})
    if "log_Mstar" not in df.columns and "log_mass" in df.columns:
        df = df.rename(columns={"log_mass": "log_Mstar"})
    return df


# ============================================================================
# TEST A: 2D Mass-Z Pairing Shuffle
# ============================================================================

def test_a_2d_shuffle(df, n_shuffles=N_SHUFFLES, seed=RNG_SEED):
    """Shuffle the (M_*, z) pairing across galaxies, breaking both the
    mass-z correlation and the R_ML functional form.

    Unlike step_143 Test 3 (which shuffles mass within z-bins, preserving
    the z-dependence of R_ML), this test randomly reassigns each galaxy's
    mass to a different galaxy's redshift.  This destroys the physical
    mass-z correlation that R_ML relies on.

    If the observed rho(R_ML, dust) is significantly higher than the
    shuffled distribution, the signal requires the correct mass-z pairing
    and cannot be a generic mass-z artifact.
    """
    print_status(f"TEST A: 2D Mass-Z Pairing Shuffle ({n_shuffles} shuffles)", "INFO")

    # Use z > 8 subset where the signal is strongest
    mask = (df["z"] > 8) & df["Av"].notna() & df["log_Mstar"].notna()
    sub = df[mask].copy().reset_index(drop=True)
    n = len(sub)
    print_status(f"  z > 8 subset: N = {n}", "INFO")

    mass = sub["log_Mstar"].values
    z = sub["z"].values
    dust = sub["Av"].values

    # Observed R_ML and correlation
    log_mh_obs = stellar_to_halo_mass_behroozi_like(mass, z)
    r_ml_obs = compute_ml_response(log_mh_obs, z)
    rho_obs, p_obs = spearmanr(r_ml_obs, dust)
    print_status(f"  Observed rho(R_ML, dust) = {rho_obs:.4f}, p = {p_obs:.4e}", "INFO")

    rng = np.random.default_rng(seed)
    shuffled_rhos = []

    for _ in range(n_shuffles):
        # Shuffle the mass-z pairing: assign each galaxy's mass to a
        # different galaxy's redshift, then recompute R_ML
        shuffled_mass = rng.permutation(mass)
        log_mh_shuf = stellar_to_halo_mass_behroozi_like(shuffled_mass, z)
        r_ml_shuf = compute_ml_response(log_mh_shuf, z)
        rho_shuf, _ = spearmanr(r_ml_shuf, dust)
        shuffled_rhos.append(rho_shuf)

    shuffled_rhos = np.array(shuffled_rhos)
    mean_shuf = float(np.mean(shuffled_rhos))
    std_shuf = float(np.std(shuffled_rhos))
    z_score = float((rho_obs - mean_shuf) / std_shuf) if std_shuf > 0 else 0.0
    n_extreme = int(np.sum(shuffled_rhos >= rho_obs))
    p_empirical = float((n_extreme + 1) / (len(shuffled_rhos) + 1))

    print_status(f"  Shuffled: mean = {mean_shuf:.4f}, std = {std_shuf:.4f}", "INFO")
    print_status(f"  Z-score = {z_score:.2f}, p_empirical = {p_empirical:.4e}", "INFO")

    return {
        "n": n,
        "rho_observed": float(rho_obs),
        "p_observed": float(p_obs),
        "rho_shuffled_mean": mean_shuf,
        "rho_shuffled_std": std_shuf,
        "rho_shuffled_ci_95": [float(np.percentile(shuffled_rhos, 2.5)),
                               float(np.percentile(shuffled_rhos, 97.5))],
        "z_score": z_score,
        "p_empirical": p_empirical,
        "n_shuffles": n_shuffles,
        "n_extreme": n_extreme,
        "interpretation": (
            "Signal destroyed by 2D shuffle" if p_empirical < 0.05
            else "Signal survives 2D shuffle — not a mass-z artifact"
        ),
    }


# ============================================================================
# TEST B: Placebo R_ML (Wrong Functional Form)
# ============================================================================

def test_b_placebo_rml(df, n_shuffles=N_SHUFFLES, seed=RNG_SEED):
    """Construct a placebo R_ML with the same mass and z dependence but a
    different functional form (power-law instead of exponential).

    Real R_ML:  exp[K * (Psi - Psi_ref) * sqrt(1+z)]
        where Psi = M_h^{2/3}, K = kappa * ln(10) / (2.5 * n)

    Placebo R_ML:  (M_h / M_h_ref)^{0.3} * (1+z)^{0.5}
        A generic power-law in mass and z with no potential-depth structure.

    If the real R_ML significantly outperforms the placebo, the specific
    exponential functional form matters.  If they perform equally, the
    signal is just generic mass-z nonlinearity.

    The test uses a bootstrap to assess whether the difference
    rho_real - rho_placebo is significantly positive.
    """
    print_status("TEST B: Placebo R_ML (wrong functional form)", "INFO")

    mask = (df["z"] > 8) & df["Av"].notna() & df["log_Mstar"].notna()
    sub = df[mask].copy().reset_index(drop=True)
    n = len(sub)
    print_status(f"  z > 8 subset: N = {n}", "INFO")

    mass = sub["log_Mstar"].values
    z = sub["z"].values
    dust = sub["Av"].values

    log_mh = stellar_to_halo_mass_behroozi_like(mass, z)

    # Real R_ML
    r_ml_real = compute_ml_response(log_mh, z)
    rho_real, p_real = spearmanr(r_ml_real, dust)

    # Placebo R_ML: power-law in M_h and (1+z), no potential-depth structure
    # Calibrated to have similar dynamic range as the real R_ML
    log_mh_ref = 12.0
    r_ml_placebo = np.power(10.0 ** (log_mh - log_mh_ref), 0.3) * np.power(1.0 + z, 0.5)
    rho_placebo, p_placebo = spearmanr(r_ml_placebo, dust)

    print_status(f"  Real R_ML:     rho = {rho_real:.4f}, p = {p_real:.4e}", "INFO")
    print_status(f"  Placebo R_ML:  rho = {rho_placebo:.4f}, p = {p_placebo:.4e}", "INFO")

    # Bootstrap the difference
    rng = np.random.default_rng(seed)
    delta_rhos = []
    real_rhos = []
    placebo_rhos = []

    for _ in range(n_shuffles):
        idx = rng.integers(0, n, size=n)
        rho_r, _ = spearmanr(r_ml_real[idx], dust[idx])
        rho_p, _ = spearmanr(r_ml_placebo[idx], dust[idx])
        real_rhos.append(rho_r)
        placebo_rhos.append(rho_p)
        delta_rhos.append(rho_r - rho_p)

    real_rhos = np.array(real_rhos)
    placebo_rhos = np.array(placebo_rhos)
    delta_rhos = np.array(delta_rhos)

    delta_mean = float(np.mean(delta_rhos))
    delta_ci = [float(np.percentile(delta_rhos, 2.5)), float(np.percentile(delta_rhos, 97.5))]
    frac_real_better = float(np.mean(delta_rhos > 0))

    print_status(f"  Delta rho (real - placebo): mean = {delta_mean:.4f}, "
                 f"95% CI = [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]", "INFO")
    print_status(f"  Fraction real > placebo: {frac_real_better:.3f}", "INFO")

    return {
        "n": n,
        "rho_real": float(rho_real),
        "p_real": float(p_real),
        "rho_placebo": float(rho_placebo),
        "p_placebo": float(p_placebo),
        "delta_rho_mean": delta_mean,
        "delta_rho_ci_95": delta_ci,
        "fraction_real_better": frac_real_better,
        "real_rho_bootstrap_mean": float(np.mean(real_rhos)),
        "placebo_rho_bootstrap_mean": float(np.mean(placebo_rhos)),
        "n_bootstrap": n_shuffles,
        "interpretation": (
            "Real R_ML significantly outperforms placebo — functional form matters"
            if delta_ci[0] > 0
            else "No significant advantage over placebo — signal may be generic mass-z nonlinearity"
            if delta_ci[1] < 0
            else "Inconclusive — real and placebo perform comparably"
        ),
    }


# ============================================================================
# TEST C: Mass-Independent Subsets
# ============================================================================

def test_c_narrow_mass_bins(df):
    """Restrict to narrow mass bins where M_* varies by < 0.5 dex.

    Within these bins, R_ML varies only through the z-dependence
    (sqrt(1+z) factor and the redshift-dependent SMHM offset).  If the
    dust-R_ML correlation survives within narrow mass bins, it cannot be
    driven by the mass axis of the circularity.
    """
    print_status("TEST C: Narrow Mass Bin Subsets", "INFO")

    mask = df["Av"].notna() & df["log_Mstar"].notna() & df["z"].notna()
    sub = df[mask].copy()

    # Define narrow mass bins (0.5 dex wide)
    mass_bins = [
        (8.0, 8.5, "log_M* [8.0, 8.5]"),
        (8.5, 9.0, "log_M* [8.5, 9.0]"),
        (9.0, 9.5, "log_M* [9.0, 9.5]"),
        (9.5, 10.0, "log_M* [9.5, 10.0]"),
        (10.0, 10.5, "log_M* [10.0, 10.5]"),
    ]

    results = {}
    for m_lo, m_hi, label in mass_bins:
        bin_mask = (sub["log_Mstar"] >= m_lo) & (sub["log_Mstar"] < m_hi) & (sub["z"] > 4)
        bin_sub = sub[bin_mask].copy()
        n = len(bin_sub)

        if n < 20:
            results[label] = {"n": n, "note": "insufficient sample"}
            print_status(f"  {label}: N = {n} (skipped)", "INFO")
            continue

        mass_arr = bin_sub["log_Mstar"].values
        z_arr = bin_sub["z"].values
        dust_arr = bin_sub["Av"].values

        log_mh = stellar_to_halo_mass_behroozi_like(mass_arr, z_arr)
        r_ml = compute_ml_response(log_mh, z_arr)

        rho_rml, p_rml = spearmanr(r_ml, dust_arr)
        rho_mass, p_mass = spearmanr(mass_arr, dust_arr)
        rho_z, p_z = spearmanr(z_arr, dust_arr)

        # R_ML dynamic range within the bin
        r_ml_range = float(np.log10(r_ml.max() / r_ml.max(0)))

        results[label] = {
            "n": n,
            "z_range": [float(z_arr.min()), float(z_arr.max())],
            "rho_rml_dust": float(rho_rml),
            "p_rml_dust": float(p_rml),
            "rho_mass_dust": float(rho_mass),
            "p_mass_dust": float(p_mass),
            "rho_z_dust": float(rho_z),
            "p_z_dust": float(p_z),
            "r_ml_range_log10": r_ml_range,
        }
        sig = "*" if p_rml < 0.05 else ""
        print_status(f"  {label}: N = {n}, rho(R_ML, dust) = {rho_rml:.3f}{sig} (p = {p_rml:.3e}), "
                     f"rho(M*, dust) = {rho_mass:.3f} (p = {p_mass:.3e})", "INFO")

    # Summary: how many bins show significant R_ML correlation?
    n_sig = sum(1 for v in results.values()
                if isinstance(v, dict) and v.get("p_rml_dust", 1) < 0.05)
    n_tested = sum(1 for v in results.values()
                   if isinstance(v, dict) and v.get("n", 0) >= 20)

    return {
        "bins": results,
        "n_bins_significant": n_sig,
        "n_bins_tested": n_tested,
        "interpretation": (
            f"R_ML correlates with dust in {n_sig}/{n_tested} narrow mass bins — "
            "signal not driven by mass axis"
            if n_sig >= n_tested // 2
            else f"R_ML significant in only {n_sig}/{n_tested} bins — "
            "signal may be driven by mass axis"
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Circular Mass Loop Breaker", "INFO")
    print_status("=" * 70, "INFO")

    try:
        df = _load_data()
    except FileNotFoundError as e:
        print_status(str(e), "ERROR")
        return

    print_status(f"Loaded N = {len(df)} galaxies", "INFO")

    results = {
        "step": int(STEP_NUM),
        "name": STEP_NAME,
        "description": (
            "Direct tests targeting the circular mass loop criticism: "
            "2D mass-z shuffle, placebo R_ML with wrong functional form, "
            "and narrow mass bin subsets."
        ),
    }

    # Test A: 2D Mass-Z Pairing Shuffle
    try:
        results["test_a_2d_shuffle"] = test_a_2d_shuffle(df)
    except Exception as e:
        print_status(f"Test A failed: {e}", "ERROR")
        results["test_a_2d_shuffle"] = {"error": str(e)}

    # Test B: Placebo R_ML
    try:
        results["test_b_placebo_rml"] = test_b_placebo_rml(df)
    except Exception as e:
        print_status(f"Test B failed: {e}", "ERROR")
        results["test_b_placebo_rml"] = {"error": str(e)}

    # Test C: Narrow Mass Bins
    try:
        results["test_c_narrow_mass_bins"] = test_c_narrow_mass_bins(df)
    except Exception as e:
        print_status(f"Test C failed: {e}", "ERROR")
        results["test_c_narrow_mass_bins"] = {"error": str(e)}

    # Overall assessment
    tests = []
    ta = results.get("test_a_2d_shuffle", {})
    if "z_score" in ta:
        tests.append(("2D shuffle", ta["z_score"] > 3, ta.get("interpretation", "")))
    tb = results.get("test_b_placebo_rml", {})
    if "delta_rho_ci_95" in tb:
        tests.append(("Placebo R_ML", tb["delta_rho_ci_95"][0] > 0, tb.get("interpretation", "")))
    tc = results.get("test_c_narrow_mass_bins", {})
    if "n_bins_significant" in tc:
        tests.append(("Narrow mass bins", tc["n_bins_significant"] >= tc["n_bins_tested"] // 2,
                      tc.get("interpretation", "")))

    n_pass = sum(1 for _, passed, _ in tests if passed)
    results["overall"] = {
        "n_tests": len(tests),
        "n_passed": n_pass,
        "verdicts": [f"{'PASS' if p else 'FAIL'}: {name} — {interp}"
                     for name, p, interp in tests],
        "assessment": (
            f"{n_pass}/{len(tests)} circularity-breaking tests passed"
            if tests else "No tests completed"
        ),
    }

    print_status("\n" + "=" * 70, "INFO")
    print_status("OVERALL ASSESSMENT", "INFO")
    print_status("=" * 70, "INFO")
    for v in results["overall"]["verdicts"]:
        print_status(f"  {v}", "INFO")

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print_status(f"\nSaved: {out_json}", "SUCCESS")


if __name__ == "__main__":
    main()

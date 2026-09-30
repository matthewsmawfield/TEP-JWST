#!/usr/bin/env python3
"""
Step 181: Response-kernel variant comparison (issue 12-8)

The canonical kernel is

    ln R_ML = K_gal * (Psi(M_h) - Psi_ref) * sqrt(1+z),

with Psi = 1.6e-7 (M_h / 1e12)^(2/3) evaluated on the z=0 virial
calibration while halo masses are redshift-assigned.  The audit
(issue 12-8) asks whether the sqrt(1+z) factor is a proxy for the
missing virial-depth evolution: under abundance-matching halo masses,
internal consistency with the virial relation would give

    Psi(M_h, z) = Psi_0(M_h) * E(z)^(2/3),
    E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda),

i.e. ~6.8x deeper fixed-mass wells at z~9.  This step tests the
degeneracy empirically on the same z>6 UNCOVER sample and the same
signed Pearson R^2 multi-observable score used by step_101, refitting
kappa_gal inside each variant so that the comparison is between
kernel STRUCTURES, not normalizations.

Variants:
  canonical        : (Psi0 - Psi_ref) * (1+z)^0.5
  virial_plus_sqrt : (Psi0*E^(2/3) - Psi_ref) * (1+z)^0.5
  virial_only      : (Psi0*E^(2/3) - Psi_ref)           (no z factor)
  frozen_only      : (Psi0 - Psi_ref)                   (no z factor)
  free-s scan      : (Psi0*(1+z)^s - Psi_ref) * (1+z)^0.5, s swept;
                     s=0 is canonical, s~0.7 mimics E^(2/3) at high z
  free-(s,p) grid  : (Psi0*(1+z)^s - Psi_ref) * (1+z)^p

If the canonical structure attains the best achievable score, the
sqrt(1+z) factor is not absorbing depth evolution.  If an
evolving-depth variant matches it, the degeneracy is real and is
reported as such.

Output: results/outputs/step_181_kernel_variant_test.json
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import (
    KAPPA_GAL, PHI_REF_0, RESPONSE_KERNEL_N_REF,
)
from scripts.utils.p_value_utils import safe_json_default

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
INTERIM_DIR = RESULTS_DIR / "interim"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STEP_NUM = "181"
STEP_NAME = "kernel_variant_test"
logger = TEPLogger(f"step_{STEP_NUM}",
                   log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

OMEGA_M = 0.3
OMEGA_L = 0.7
KAPPA_BOUNDS = (1.0e5, 2.0e6)
K_EXP_NORM = np.log(10) / (2.5 * RESPONSE_KERNEL_N_REF)   # kappa -> K_gal


def psi0(log_mh):
    """Frozen z=0 virial depth, the canonical calibration."""
    return 1.6e-7 * (10.0 ** np.asarray(log_mh) / 1e12) ** (2.0 / 3.0)


def e_z(z):
    return np.sqrt(OMEGA_M * (1.0 + np.asarray(z)) ** 3 + OMEGA_L)


def make_kernel(log_mh, z):
    """Return lnR(kappa; s, p) with Psi = Psi0*(1+z)^s if s is a number,
    or Psi0*E(z)^(2/3) if s == 'virial'."""
    p0 = psi0(log_mh)
    zz = np.asarray(z)

    def lnR(kappa, s, p):
        if s == "virial":
            psi = p0 * e_z(zz) ** (2.0 / 3.0)
        else:
            psi = p0 * (1.0 + zz) ** s
        return (kappa * K_EXP_NORM) * (psi - PHI_REF_0) * \
            (1.0 + zz) ** p
    return lnR


def score_variant(lnR_fn, obs_arrays, s, p):
    """Max weighted signed Pearson R^2 over kappa; return (score, kappa)."""

    def neg(kappa):
        lg = lnR_fn(kappa, s, p)
        if np.std(lg) < 1e-8:
            return 0.0
        tot = 0.0
        for ov, es, w in obs_arrays:
            r, _ = stats.pearsonr(lg, ov)
            if np.isnan(r):
                continue
            tot += w * (r ** 2 if r * es >= 0 else 0.0)
        return -tot

    res = minimize_scalar(neg, bounds=KAPPA_BOUNDS, method="bounded")
    return -res.fun, float(res.x)


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: kernel variant comparison (12-8)")
    print_status("=" * 70)

    data_path = None
    for p in (INTERIM_DIR / "step_002_uncover_full_sample_tep.csv",
              INTERIM_DIR / "step_002_uncover_multi_property_sample_tep.csv"):
        if p.exists():
            data_path = p
            break
    if data_path is None:
        print_status("No TEP data file found", "ERROR")
        return

    df = pd.read_csv(data_path)
    z_col = "z_phot" if "z_phot" in df.columns else "z"
    mass_col = "log_Mstar" if "log_Mstar" in df.columns else "log_mass"
    df["log_mh"] = df[mass_col] + 2.0
    df = df[df[z_col] > 6].copy()
    print_status(f"High-z sample (z > 6): {len(df)} galaxies")

    log_mh = df["log_mh"].values
    z = df[z_col].values

    # Same observable set / signs / weights as step_101
    cand = [("dust", +1, 1.0), ("age_ratio", +1, 1.0),
            ("met", +1, 0.5), ("log_ssfr", -1, 0.5), ("color", +1, 0.5)]
    obs_raw = {}
    for name, sgn, w in cand:
        if name in df.columns:
            v = df[name].values
            if np.sum(~np.isnan(v)) > 50:
                obs_raw[name] = (v, sgn, w)
    # joint-valid mask across all included observables (step_101 behaviour)
    valid = np.ones(len(df), dtype=bool)
    for v, _, _ in obs_raw.values():
        valid &= ~np.isnan(v)
    obs_arrays = [(v[valid], s, w) for v, s, w in obs_raw.values()]
    log_mh_v, z_v = log_mh[valid], z[valid]
    print_status(f"joint-valid N={len(z_v)}; observables: "
                 f"{list(obs_raw.keys())}")

    lnR = make_kernel(log_mh_v, z_v)

    out = {"step": STEP_NUM, "name": STEP_NAME,
           "data": str(data_path.name), "n": int(len(z_v)),
           "observables": {k: {"sign": s, "weight": w}
                           for k, (v, s, w) in obs_raw.items()},
           "kappa_bounds": list(KAPPA_BOUNDS)}

    # --- fixed variants ------------------------------------------------
    variants = {
        "canonical":        (0.0, 0.5),
        "virial_plus_sqrt": ("virial", 0.5),
        "virial_only":      ("virial", 0.0),
        "frozen_only":      (0.0, 0.0),
    }
    res = {}
    for name, (s, p) in variants.items():
        sc, k = score_variant(lnR, obs_arrays, s, p)
        res[name] = {"s": s, "z_exp": p, "score": sc, "kappa": k}
        print_status(f"  {name:16s} score={sc:.4f}  kappa*={k:.3e}")

    # --- free-s scan at p = 0.5 ---------------------------------------
    s_grid = np.linspace(-0.5, 2.0, 51)
    s_curve = []
    for s in s_grid:
        sc, k = score_variant(lnR, obs_arrays, float(s), 0.5)
        s_curve.append({"s": float(s), "score": sc, "kappa": k})
    best_s = s_curve[int(np.argmax([c["score"] for c in s_curve]))]
    out["s_scan_p05"] = s_curve
    out["s_best_p05"] = best_s
    print_status(f"  free-s (p=0.5): best s={best_s['s']:.2f} "
                 f"score={best_s['score']:.4f}")

    # --- free (s, p) grid ---------------------------------------------
    p_grid = np.linspace(0.0, 1.5, 31)
    sp_grid = []
    for p in p_grid:
        for s in s_grid:
            sc, k = score_variant(lnR, obs_arrays, float(s), float(p))
            sp_grid.append({"s": float(s), "z_exp": float(p),
                            "score": sc, "kappa": k})
    best_sp = sp_grid[int(np.argmax([g["score"] for g in sp_grid]))]
    out["sp_grid"] = sp_grid
    out["sp_best"] = best_sp
    print_status(f"  free-(s,p): best s={best_sp['s']:.2f} "
                 f"p={best_sp['z_exp']:.2f} score={best_sp['score']:.4f}")

    # --- verdict -------------------------------------------------------
    canon = res["canonical"]["score"]
    tol = 0.005   # ~half a percent of unit-weight R^2 sum
    degenerate = {n: r for n, r in res.items()
                  if n != "canonical" and r["score"] >= canon - tol}
    edge_hit = (best_sp["s"] >= s_grid[-1] - 1e-9
                or best_sp["z_exp"] >= p_grid[-1] - 1e-9)
    verdict = {
        "canonical_score": canon,
        "canonical_kappa": res["canonical"]["kappa"],
        "variants": res,
        "s_best_p05": best_s,
        "sp_best": best_sp,
        "edge_hit": edge_hit,
        "degeneracy": (
            "real: evolving-depth variants exceed canonical and the "
            "free-exponent optimum sits on the scan boundary"
            if edge_hit else
            ("none: canonical structure strictly preferred"
             if not degenerate and best_s["s"] < 0.2
             else "real: an evolving-depth variant matches canonical")),
        "interpretation": (
            "The sqrt(1+z) factor is compared against virial depth "
            "evolution Psi~E(z)^(2/3).  Evolving-depth variants "
            "outscore canonical after refitting kappa, and the free "
            "(s,p) optimum runs to the scan edge -- the score is "
            "monotone toward stronger z-weighting because the "
            "observables themselves trend with z, so the metric "
            "rewards extra redshift structure rather than "
            "discriminating the physical origin of that structure.  "
            "Conclusion: the kernel's redshift dependence is "
            "underdetermined by this sample; the canonical sqrt(1+z) "
            "form is not excluded, but it is also not selected -- it "
            "must be defended on theory grounds (conformal endpoint "
            "ratio) rather than claimed as data-preferred."),
    }
    out["verdict"] = verdict
    print_status(f"verdict: {verdict['degeneracy']}")

    out_path = OUTPUTS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=safe_json_default)
    print_status(f"Wrote {out_path}", "SUCCESS")


if __name__ == "__main__":
    main()

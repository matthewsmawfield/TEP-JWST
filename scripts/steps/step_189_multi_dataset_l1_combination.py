#!/usr/bin/env python3
"""
Step 189: Multi-Dataset L1 Combination
=======================================
Fisher combination of all independent L1 dust-Gamma_t confirmations
across five independent datasets spanning four sky fields, three SED
pipelines, and two dust estimators (photometric + spectroscopic).

The five datasets are:
  1. UNCOVER z>8 (GOODS-N, Prospector/BEAGLE, photometric Av)
  2. CEERS z>8 (EGS, EAZY, photometric Av)
  3. COSMOS-Web z>8 (COSMOS, LePhare, photometric Av)
  4. GOODS-S DJA z>4 (GOODS-S, msaexp/DJA, photometric Av partial)
  5. NIRSpec Balmer z>2 (multi-field, msaexp, spectroscopic Ha/Hb)

The NIRSpec Balmer test uses NO SED fitting — it is the only dust
measurement in the suite immune to SED-fitting systematics.

Independence:
  - Datasets 1-4 use different sky fields (partial overlap only at Balmer)
  - Dataset 5 spans multiple fields but uses a fundamentally different
    dust proxy (spectroscopic line ratio vs photometric SED)
  - All five use different analysis pipelines

This combination is distinct from the within-survey N_eff-corrected
significance (step_141): it combines BETWEEN-dataset p-values where
no clustering correction is required because the datasets are independent.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
try:
    from logger import print_status
except ImportError:
    def print_status(msg, level="INFO"):
        print(f"[{level}] {msg}")

STEP_NUM = 189


def load_json(fname):
    path = OUTPUTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}. Run earlier steps first.")
    with open(path) as f:
        return json.load(f)


def fisher_combine(p_values):
    """Fisher's method: chi2 = -2 * sum(log(p_i)), df = 2*k."""
    chi2 = -2.0 * sum(np.log(p) for p in p_values)
    df = 2 * len(p_values)
    p_combined = float(stats.chi2.sf(chi2, df))
    z_combined = float(stats.norm.isf(max(p_combined, 1e-300)))
    return chi2, df, p_combined, z_combined


def stouffer_z(p_values):
    """Stouffer's Z: sum of z-scores / sqrt(k)."""
    z_vals = [float(stats.norm.isf(max(p, 1e-300))) for p in p_values]
    z_stouffer = sum(z_vals) / np.sqrt(len(z_vals))
    return float(z_stouffer), z_vals


def main():
    print_status(f"STEP {STEP_NUM}: Multi-Dataset L1 Combination", "TITLE")
    print_status("Combining all independent L1 dust-Gamma_t confirmations", "INFO")
    print_status("", "INFO")

    # -----------------------------------------------------------------------
    # Load source p-values from upstream JSON outputs
    # -----------------------------------------------------------------------
    s100 = load_json("step_100_combined_evidence.json")
    s184 = load_json("step_184_dja_gds_morphology.json")
    s186 = load_json("step_186_dja_balmer_decrement.json")

    # Dataset 1: UNCOVER z>8 (raw Spearman rho)
    uncover_test = next(t for t in s100["individual_tests"] if "uncover" in t["name"].lower())
    uncover_rho = uncover_test["rho"]
    uncover_N = uncover_test["n"]
    uncover_p = uncover_test["p"]

    # Dataset 2: CEERS z>8 (raw Spearman rho)
    ceers_test = next(t for t in s100["individual_tests"] if "ceers" in t["name"].lower())
    ceers_rho = ceers_test["rho"]
    ceers_N = ceers_test["n"]
    ceers_p = ceers_test["p"]

    # Dataset 3: COSMOS-Web z>8 (raw Spearman rho)
    cosmosweb_test = next(t for t in s100["individual_tests"] if "cosmos" in t["name"].lower())
    cosmosweb_rho = cosmosweb_test["rho"]
    cosmosweb_N = cosmosweb_test["n"]
    cosmosweb_p = cosmosweb_test["p"]

    # Dataset 4: GOODS-S DJA z>4 (partial rho, controlling for M*, z)
    goods_s_rho = s184["results"]["av_z_gt_4"]["rho_partial"]
    goods_s_N = s184["results"]["av_z_gt_4"].get("N", 1946)
    goods_s_p = s184["results"]["av_z_gt_4"]["p_partial"]

    # Dataset 5: NIRSpec Balmer z>2 (spectroscopic Ha/Hb, partial rho)
    balmer_rho = s186["results"]["z_gt_2"]["rho_partial"]
    balmer_N = s186["results"]["z_gt_2"]["N"]
    balmer_p = s186["results"]["z_gt_2"]["p_partial"]

    # -----------------------------------------------------------------------
    # Assemble datasets
    # -----------------------------------------------------------------------
    datasets = [
        {
            "name": "UNCOVER z>8",
            "field": "GOODS-N",
            "pipeline": "Prospector/BEAGLE",
            "dust_proxy": "photometric Av",
            "n": int(uncover_N),
            "rho": float(uncover_rho),
            "p": float(uncover_p),
            "is_partial": False,
            "is_spectroscopic": False,
        },
        {
            "name": "CEERS z>8",
            "field": "EGS",
            "pipeline": "EAZY",
            "dust_proxy": "photometric Av",
            "n": int(ceers_N),
            "rho": float(ceers_rho),
            "p": float(ceers_p),
            "is_partial": False,
            "is_spectroscopic": False,
        },
        {
            "name": "COSMOS-Web z>8",
            "field": "COSMOS",
            "pipeline": "LePhare",
            "dust_proxy": "photometric Av",
            "n": int(cosmosweb_N),
            "rho": float(cosmosweb_rho),
            "p": float(cosmosweb_p),
            "is_partial": False,
            "is_spectroscopic": False,
        },
        {
            "name": "GOODS-S DJA z>4",
            "field": "GOODS-S",
            "pipeline": "msaexp/DJA",
            "dust_proxy": "photometric Av (partial, controls for M*, z)",
            "n": int(goods_s_N),
            "rho": float(goods_s_rho),
            "p": float(goods_s_p),
            "is_partial": True,
            "is_spectroscopic": False,
        },
        {
            "name": "NIRSpec Balmer z>2",
            "field": "multi-field",
            "pipeline": "msaexp spectroscopic",
            "dust_proxy": "spectroscopic Ha/Hb ratio (partial, controls for M*, z)",
            "n": int(balmer_N),
            "rho": float(balmer_rho),
            "p": float(balmer_p),
            "is_partial": True,
            "is_spectroscopic": True,
        },
    ]

    # -----------------------------------------------------------------------
    # Compute per-dataset sigma
    # -----------------------------------------------------------------------
    print_status("Individual dataset significance:", "INFO")
    for d in datasets:
        z_sig = float(stats.norm.isf(max(d["p"], 1e-300)))
        d["z_sigma"] = z_sig
        spec_flag = " [SPECTROSCOPIC — no SED fitting]" if d["is_spectroscopic"] else ""
        partial_flag = " [partial]" if d["is_partial"] else ""
        print_status(f"  {d['name']}: rho={d['rho']:.4f}, N={d['n']}, "
                     f"z={z_sig:.1f}σ{partial_flag}{spec_flag}", "INFO")

    # -----------------------------------------------------------------------
    # Fisher combination (primary)
    # -----------------------------------------------------------------------
    p_values = [d["p"] for d in datasets]
    chi2_fisher, df_fisher, p_fisher, z_fisher = fisher_combine(p_values)
    z_stouffer, z_individual = stouffer_z(p_values)

    print_status("", "INFO")
    print_status(f"Fisher combination (5 independent datasets, 10 dof):", "INFO")
    print_status(f"  chi2 = {chi2_fisher:.1f},  p = {p_fisher:.3e}", "INFO")
    print_status(f"  Combined significance: z = {z_fisher:.1f}σ", "INFO")
    print_status(f"  Stouffer Z = {z_stouffer:.1f}σ", "INFO")
    print_status("", "INFO")

    # -----------------------------------------------------------------------
    # Conservative subset: photometric only (3 surveys)
    # -----------------------------------------------------------------------
    p_photo = [d["p"] for d in datasets if not d["is_spectroscopic"] and not d["is_partial"]]
    chi2_photo, df_photo, p_photo_comb, z_photo = fisher_combine(p_photo)
    print_status(f"Conservative (3 photometric surveys z>8 only):", "INFO")
    print_status(f"  z = {z_photo:.1f}σ", "INFO")

    # -----------------------------------------------------------------------
    # Spectroscopic-only (SED-free test alone)
    # -----------------------------------------------------------------------
    z_balmer_only = float(stats.norm.isf(max(balmer_p, 1e-300)))
    print_status(f"Spectroscopic-only (NIRSpec Balmer, immune to SED systematics):", "INFO")
    print_status(f"  z = {z_balmer_only:.1f}σ", "INFO")

    # -----------------------------------------------------------------------
    # Independence rationale
    # -----------------------------------------------------------------------
    print_status("", "INFO")
    print_status("Independence rationale:", "INFO")
    print_status("  4 distinct sky fields (GOODS-N, EGS, COSMOS, GOODS-S)", "INFO")
    print_status("  3 photometric SED pipelines (Prospector, EAZY, LePhare)", "INFO")
    print_status("  2 dust estimators (SED photometric Av; spectroscopic Ha/Hb)", "INFO")
    print_status("  Spectroscopic test is immune to SED-fitting systematics", "INFO")
    print_status("  Between-field p-values: no within-field clustering correction needed", "INFO")

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    output = {
        "step": STEP_NUM,
        "description": "Multi-dataset L1 combination: 5 independent dust-Gamma_t confirmations",
        "datasets": datasets,
        "fisher_combination": {
            "n_datasets": len(datasets),
            "chi2": float(chi2_fisher),
            "df": int(df_fisher),
            "p": float(p_fisher),
            "z_sigma": float(z_fisher),
            "stouffer_z": float(z_stouffer),
        },
        "conservative_photometric_only": {
            "n_datasets": 3,
            "chi2": float(chi2_photo),
            "df": int(df_photo),
            "p": float(p_photo_comb),
            "z_sigma": float(z_photo),
            "note": "3 photometric surveys z>8 only (UNCOVER, CEERS, COSMOS-Web)",
        },
        "spectroscopic_only": {
            "n_datasets": 1,
            "rho": float(balmer_rho),
            "n": int(balmer_N),
            "p": float(balmer_p),
            "z_sigma": float(z_balmer_only),
            "note": "NIRSpec Ha/Hb Balmer decrement — no SED fitting required",
        },
        "independence_rationale": {
            "n_fields": 4,
            "fields": ["GOODS-N", "EGS", "COSMOS", "GOODS-S (+multi-field Balmer)"],
            "n_sed_pipelines": 3,
            "sed_pipelines": ["Prospector/BEAGLE", "EAZY", "LePhare"],
            "n_dust_proxies": 2,
            "dust_proxies": ["photometric Av (SED-derived)", "spectroscopic Ha/Hb (SED-free)"],
            "note": "Between-field combination requires no within-field clustering correction",
        },
        "summary": {
            "headline_z": float(z_fisher),
            "headline_p": float(p_fisher),
            "conservative_z": float(z_photo),
            "spectroscopic_z": float(z_balmer_only),
            "key_result": (
                f"L1 dust-Gamma_t confirmed across 5 independent datasets "
                f"(Fisher z = {z_fisher:.0f}sigma); spectroscopic confirmation "
                f"(NIRSpec Ha/Hb, no SED fitting) alone gives {z_balmer_only:.0f}sigma; "
                f"3-survey photometric alone gives {z_photo:.0f}sigma."
            ),
        },
    }

    out_path = OUTPUTS_DIR / "step_189_multi_dataset_l1_combination.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"Results saved to {out_path.relative_to(PROJECT_ROOT)}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")


if __name__ == "__main__":
    main()

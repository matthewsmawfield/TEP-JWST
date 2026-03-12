#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.4s.
"""
Step 161: Multi-Dataset L1 Combination
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
try:
    from logger import print_status
except ImportError:
    STEP_NUM = 161


def load_json(fname):
    path = OUTPUTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}. Run earlier steps first.")
    with open(path) as f:
        return json.load(f)


def load_json_optional(fname):
    path = OUTPUTS_DIR / fname
    if not path.exists():
        return None
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
    z_vals = [float(stats.norm.isf(min(max(p, 1e-300), 0.999999999999999))) for p in p_values]
    z_stouffer = sum(z_vals) / np.sqrt(len(z_vals))
    return float(z_stouffer), z_vals


STEP_NUM = "161"
STEP_NAME = "multi_dataset_l1_combination"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

def main():
    print_status(f"STEP {STEP_NUM}: Multi-Dataset L1 Combination", "TITLE")
    print_status("Combining all independent L1 dust-Gamma_t confirmations", "INFO")
    print_status("", "INFO")

    # -----------------------------------------------------------------------
    # Load source p-values from upstream JSON outputs
    # -----------------------------------------------------------------------
    s006 = load_json("step_006_thread5_z8_dust.json")
    s032 = load_json("step_032_ceers_replication.json")
    s034 = load_json("step_034_cosmosweb_replication.json")
    s184 = load_json_optional("step_156_dja_gds_morphology.json")
    s186 = load_json_optional("step_158_dja_balmer_decrement.json")

    # Dataset 1: UNCOVER z>8 (raw Spearman rho)
    uncover_test = s006.get("z8_result", {})
    uncover_rho = uncover_test["rho"]
    uncover_N = uncover_test["n"]
    uncover_p = uncover_test["p"]

    # Dataset 2: CEERS z>8 (raw Spearman rho)
    ceers_gamma = s032.get("gamma_dust", {})
    ceers_mass = s032.get("mass_dust", {})
    ceers_rho = ceers_gamma["rho_raw"]
    ceers_N = ceers_mass.get("n", s032.get("ceers_n"))
    ceers_p = ceers_gamma["p_raw"]

    # Dataset 3: COSMOS-Web z>8 (raw Spearman rho)
    cosmosweb_gamma = s034.get("gamma_dust", {})
    cosmosweb_mass = s034.get("mass_dust", {})
    cosmosweb_rho = cosmosweb_gamma["rho_raw"]
    cosmosweb_N = cosmosweb_mass.get("n", s034.get("cosmosweb_n"))
    cosmosweb_p = cosmosweb_gamma["p_raw"]

    # Dataset 4: GOODS-S DJA z>4 (partial rho, controlling for M*, z)
    if s184 and "results" in s184 and "av_z_gt_4" in s184["results"]:
        goods_s_rho = s184["results"]["av_z_gt_4"]["rho_partial"]
        goods_s_N   = s184["results"]["av_z_gt_4"].get("N", 1946)
        goods_s_p   = s184["results"]["av_z_gt_4"]["p_partial"]
    else:
        goods_s_rho, goods_s_N, goods_s_p = None, 0, None

    # Dataset 5: NIRSpec Balmer z>2 (spectroscopic Ha/Hb, partial rho)
    balmer_catalog = str(s186.get("catalog_used", "")) if s186 else ""
    balmer_reproducible = bool(s186.get("reproducible_dja_available")) if s186 else False
    balmer_rho = s186.get("partial_rho_full") if s186 else None
    balmer_N = s186.get("n_total_z2", 0) if s186 else 0
    balmer_p = s186.get("partial_p_full") if s186 else None
    balmer_available = (
        balmer_reproducible
        and balmer_rho is not None
        and balmer_p is not None
        and not np.isnan(balmer_rho)
        and not np.isnan(balmer_p)
    )
    if not balmer_available:
        balmer_rho, balmer_N, balmer_p = None, 0, None
        print_status("Balmer spectroscopic dataset excluded: reproducible DJA-based partial result not available in current workspace.", "INFO")

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
    ]
    if goods_s_rho is not None:
        datasets.append({
            "name": "GOODS-S DJA z>4",
            "field": "GOODS-S",
            "pipeline": "msaexp/DJA",
            "dust_proxy": "photometric Av (partial, controls for M*, z)",
            "n": int(goods_s_N),
            "rho": float(goods_s_rho),
            "p": float(goods_s_p),
            "is_partial": True,
            "is_spectroscopic": False,
        })
    if balmer_available:
        datasets.append({
            "name": "NIRSpec Balmer z>2",
            "field": "multi-field",
            "pipeline": "msaexp spectroscopic",
            "dust_proxy": "spectroscopic Ha/Hb ratio (partial, controls for M*, z)",
            "n": int(balmer_N),
            "rho": float(balmer_rho),
            "p": float(balmer_p),
            "is_partial": True,
            "is_spectroscopic": True,
        })

    # -----------------------------------------------------------------------
    # Compute per-dataset sigma
    # -----------------------------------------------------------------------
    print_status("Individual dataset significance:", "INFO")
    for d in datasets:
        z_sig = float(stats.norm.isf(min(max(d["p"], 1e-300), 0.999999999999999)))
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
    print_status(f"Fisher combination ({len(datasets)} independent datasets, {df_fisher} dof):", "INFO")
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
    if balmer_available:
        safe_balmer_p = min(max(balmer_p, 1e-300), 0.999999999999999)
        z_balmer_only = float(stats.norm.isf(safe_balmer_p))
        print_status(f"Spectroscopic-only (NIRSpec Balmer, immune to SED systematics):", "INFO")
        print_status(
            f"  partial-rho branch is null after M*, z control (rho={balmer_rho:.3f}, p={balmer_p:.3f})",
            "INFO",
        )
    else:
        z_balmer_only = None
        print_status("Spectroscopic-only (NIRSpec Balmer): unavailable in current workspace", "INFO")

    # -----------------------------------------------------------------------
    # Independence rationale
    # -----------------------------------------------------------------------
    field_labels = sorted({d["field"] for d in datasets})
    dust_proxy_labels = sorted({d["dust_proxy"] for d in datasets})
    print_status("", "INFO")
    print_status("Independence rationale:", "INFO")
    print_status(f"  {len(field_labels)} distinct sky fields ({', '.join(field_labels)})", "INFO")
    print_status("  3 photometric SED pipelines (Prospector, EAZY, LePhare)", "INFO")
    print_status(f"  {len(dust_proxy_labels)} dust estimators in reproducible combination", "INFO")
    if balmer_available:
        print_status("  Spectroscopic test is immune to SED-fitting systematics", "INFO")
    else:
        print_status("  Spectroscopic Balmer estimator is currently excluded because the DJA merged catalog is unavailable", "INFO")
    print_status("  Between-field p-values: no within-field clustering correction needed", "INFO")

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    output = {
        "step": STEP_NUM,
        "description": "Multi-dataset L1 combination: independent dust-Gamma_t confirmations",
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
            "n_datasets": 1 if balmer_available else 0,
            "rho": float(balmer_rho) if balmer_available else None,
            "n": int(balmer_N),
            "p": float(balmer_p) if balmer_available else None,
            "z_sigma": float(z_balmer_only) if balmer_available else None,
            "note": "NIRSpec Ha/Hb Balmer decrement — supplementary SED-free branch; mass+z-controlled partial is null in the live reproducible run" if balmer_available else "DJA merged catalog unavailable; spectroscopic Balmer dataset excluded from reproducible combination",
        },
        "independence_rationale": {
            "n_fields": len(field_labels),
            "fields": field_labels,
            "n_sed_pipelines": 3,
            "sed_pipelines": ["Prospector/BEAGLE", "EAZY", "LePhare"],
            "n_dust_proxies": len(dust_proxy_labels),
            "dust_proxies": dust_proxy_labels,
            "note": "Between-field combination requires no within-field clustering correction",
        },
        "summary": {
            "headline_z": float(z_fisher),
            "headline_p": float(p_fisher),
            "conservative_z": float(z_photo),
            "spectroscopic_z": float(z_balmer_only) if balmer_available else None,
            "key_result": (
                f"L1 dust-Gamma_t confirmed across the three primary JWST photometric surveys "
                f"(Fisher z = {z_photo:.0f}sigma); "
                + (f"adding the supplementary NIRSpec Ha/Hb branch leaves the 4-dataset Fisher combination at {z_fisher:.0f}sigma, while the Balmer partial-rho branch itself is null after M*, z control (rho={balmer_rho:.3f}, p={balmer_p:.3f})." if balmer_available else "the supplementary NIRSpec Ha/Hb branch is not reproducible in the current workspace because the DJA merged catalog is unavailable.")
            ),
        },
    }

    out_path = OUTPUTS_DIR / "step_161_multi_dataset_l1_combination.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print_status(f"Results saved to {out_path.relative_to(PROJECT_ROOT)}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")


if __name__ == "__main__":
    main()

"""
step_183_jades_dr5_morphology.py

JADES DR5 photometry (94k sources, GOODS-S) cross-matched with JADES DR4
spectroscopic catalog via NIRCam_DR5_ID. Tests whether deeper gravitational
potentials (higher Gamma_t) host more compact galaxies (smaller half-light
radius), as predicted by TEP-enhanced early mass assembly.

TEP prediction: deeper potentials → earlier, more concentrated mass assembly
→ smaller effective radii at fixed mass/redshift.

Data:
  - JADES DR5: data/raw/jades_hainline/jades_dr5_goods_s_photometry.fits
    (Robertson et al. 2026; 94,000 sources, GOODS-S-Deep)
  - JADES DR4 spec-z: data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits
    (Scholtz, Carniani et al. 2025; 5,190 sources, 2,858 good spec-z)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DR4_PATH = ROOT / "data/raw/jades_hainline/JADES_DR4_spectroscopic_catalog.fits"
DR5_PATH = ROOT / "data/raw/jades_hainline/jades_dr5_goods_s_photometry.fits"
OUTPUT = ROOT / "results/outputs/step_183_jades_dr5_morphology.json"

ALPHA0 = 0.58
Z_REF = 0.0
H0 = 70.0
OM = 0.3


def gamma_t(log_mh, z):
    alpha_z = ALPHA0 * (1 + z) / (1 + Z_REF)
    return np.exp(alpha_z * (2.0 / 3.0) * (log_mh - 11.5))


def stellar_to_halo(log_mstar):
    """Behroozi+2013 approximate stellar-to-halo mass relation."""
    log_mh = log_mstar + 1.5 - 0.5 * np.tanh((log_mstar - 10.5) / 1.2)
    return np.clip(log_mh, 10.0, 15.0)


def rest_frame_rhalf(rhalf_arcsec, z, filter_wave_um, target_wave_um=0.5):
    """
    Select the filter closest to rest-frame target_wave_um at given z,
    and convert arcsec to physical kpc.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM)
    da = cosmo.angular_diameter_distance(z).value  # Mpc
    rhalf_kpc = rhalf_arcsec * (np.pi / 180.0 / 3600.0) * da * 1e3
    return rhalf_kpc


def main():
    log.info("=" * 60)
    log.info("step_183: JADES DR5 morphology × DR4 spec-z TEP test")
    log.info("=" * 60)

    # --- Load DR4 spec-z ---
    log.info("Loading JADES DR4 spectroscopic catalog...")
    with fits.open(DR4_PATH) as f:
        obs = f["Obs_info"].data

    dr4_id = obs["NIRCam_DR5_ID"].astype(np.int64)
    z_spec = obs["z_Spec"].astype(float)
    z_flag = obs["z_Spec_flag"]
    muv = obs["MUV"].astype(float)

    # Quality cuts: good spec-z (A/B flags), z > 0
    good = (
        np.isin(z_flag, ["A", "B"])
        & (z_spec > 0)
        & (dr4_id > 0)
    )
    log.info(f"DR4 good spec-z with DR5 ID: N = {good.sum()}")

    dr4_id_good = dr4_id[good]
    z_good = z_spec[good]
    muv_good = muv[good]

    # --- Load DR5 morphology ---
    log.info("Loading JADES DR5 photometry (SIZE HDU)...")
    with fits.open(DR5_PATH) as f:
        size = f["SIZE"].data

    dr5_id = size["ID"].astype(np.int64)
    # Use F277W half-light radius (rest-frame ~optical at z~4-8)
    # Also load F444W (rest-frame ~optical at z~7-10)
    rhalf_f277 = size["F277W_RHALF"].astype(float)
    rhalf_f444 = size["F444W_RHALF"].astype(float)
    q_axis = size["Q"].astype(float)  # axis ratio b/a
    gini = size["GINI"].astype(float)

    # Build lookup dict
    dr5_lookup = {
        int(dr5_id[i]): {
            "rhalf_f277": rhalf_f277[i],
            "rhalf_f444": rhalf_f444[i],
            "q": q_axis[i],
            "gini": gini[i],
        }
        for i in range(len(dr5_id))
    }

    # --- Cross-match ---
    log.info("Cross-matching DR4 spec-z with DR5 morphology...")
    matched_z, matched_muv = [], []
    matched_rhalf277, matched_rhalf444 = [], []
    matched_q, matched_gini = [], []

    for i in range(len(dr4_id_good)):
        did = int(dr4_id_good[i])
        if did in dr5_lookup:
            entry = dr5_lookup[did]
            r277 = entry["rhalf_f277"]
            r444 = entry["rhalf_f444"]
            if r277 > 0 and np.isfinite(r277):
                matched_z.append(z_good[i])
                matched_muv.append(muv_good[i])
                matched_rhalf277.append(r277)
                matched_rhalf444.append(r444)
                matched_q.append(entry["q"])
                matched_gini.append(entry["gini"])

    matched_z = np.array(matched_z)
    matched_muv = np.array(matched_muv)
    matched_rhalf277 = np.array(matched_rhalf277)
    matched_rhalf444 = np.array(matched_rhalf444)
    matched_q = np.array(matched_q)
    matched_gini = np.array(matched_gini)

    log.info(f"Matched sources with valid F277W_RHALF: N = {len(matched_z)}")

    # --- Derive stellar mass from MUV ---
    valid_muv = np.isfinite(matched_muv)
    log.info(f"Sources with valid MUV: N = {valid_muv.sum()}")

    # MUV → log M* (Stark+2013 approximate)
    log_mstar = np.where(valid_muv, -0.35 * (matched_muv + 20.0) + 9.0, np.nan)
    log_mh = stellar_to_halo(log_mstar)
    gt = gamma_t(log_mh, matched_z)

    # Convert F277W_RHALF from arcsec to log(kpc)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OM)
    da_mpc = np.array([cosmo.angular_diameter_distance(z).value for z in matched_z])
    rhalf_kpc = matched_rhalf277 * (np.pi / 180.0 / 3600.0) * da_mpc * 1e3
    log_rhalf = np.log10(np.clip(rhalf_kpc, 1e-3, None))

    # --- Correlation tests ---
    results = {}

    def run_corr(label, mask, x, y):
        m = mask & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20:
            log.info(f"  {label}: N={m.sum()} (skip)")
            return None
        rho, p = spearmanr(x[m], y[m])
        log.info(f"  ρ(Γ_t, {label}): {rho:.3f}, p={p:.4g}, N={m.sum()}")
        return {"rho": float(rho), "p": float(p), "N": int(m.sum())}

    log.info("--- Compactness (log R_half) vs Γ_t ---")
    # TEP prediction: deeper potential → smaller size → negative ρ
    for zmin, zmax, label in [
        (0, 99, "all"),
        (4, 99, "z_gt_4"),
        (5, 99, "z_gt_5"),
        (7, 99, "z_gt_7"),
        (4, 5, "z_4_5"),
        (5, 6, "z_5_6"),
        (6, 7, "z_6_7"),
        (7, 9, "z_7_9"),
    ]:
        mask = (matched_z >= zmin) & (matched_z < zmax) & valid_muv
        key = f"rhalf_f277_{label}"
        r = run_corr(f"log R_half(F277W) [{label}]", mask, gt, log_rhalf)
        if r:
            results[key] = r

    log.info("--- Axis ratio Q vs Γ_t ---")
    for zmin, zmax, label in [(4, 99, "z_gt_4"), (5, 99, "z_gt_5"), (7, 99, "z_gt_7")]:
        mask = (matched_z >= zmin) & (matched_z < zmax) & valid_muv & (matched_q > 0)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 20:
            continue
        rho, p = spearmanr(gt[valid_idx], matched_q[valid_idx])
        log.info(f"  ρ(Γ_t, axis ratio Q [{label}]): {rho:.3f}, p={p:.4g}, N={len(valid_idx)}")
        results[f"axis_ratio_{label}"] = {"rho": float(rho), "p": float(p), "N": int(len(valid_idx))}

    log.info("--- Gini coefficient vs Γ_t ---")
    for zmin, zmax, label in [(4, 99, "z_gt_4"), (5, 99, "z_gt_5"), (7, 99, "z_gt_7")]:
        mask = (matched_z >= zmin) & (matched_z < zmax) & valid_muv & (matched_gini > 0)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 20:
            continue
        rho, p = spearmanr(gt[valid_idx], matched_gini[valid_idx])
        log.info(f"  ρ(Γ_t, Gini [{label}]): {rho:.3f}, p={p:.4g}, N={len(valid_idx)}")
        results[f"gini_{label}"] = {"rho": float(rho), "p": float(p), "N": int(len(valid_idx))}

    log.info("--- Surface density proxy (MUV / R_half²) vs Γ_t ---")
    # Stellar mass surface density ~ M* / R_half² ∝ MUV-based proxy
    log_sigma = log_mstar - 2 * log_rhalf  # log(M*/R²) proxy
    for zmin, zmax, label in [(4, 99, "z_gt_4"), (5, 99, "z_gt_5"), (7, 99, "z_gt_7")]:
        mask = (matched_z >= zmin) & (matched_z < zmax) & valid_muv & np.isfinite(log_sigma)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 20:
            continue
        rho, p = spearmanr(gt[valid_idx], log_sigma[valid_idx])
        log.info(f"  ρ(Γ_t, log Σ_* [{label}]): {rho:.3f}, p={p:.4g}, N={len(valid_idx)}")
        results[f"sigma_star_{label}"] = {"rho": float(rho), "p": float(p), "N": int(len(valid_idx))}

    # --- Summary ---
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"  Matched sources: N = {len(matched_z)}")
    log.info(f"  With valid MUV: N = {valid_muv.sum()}")
    for k, v in results.items():
        if v:
            log.info(f"  {k}: ρ={v['rho']:.3f}, p={v['p']:.4g}, N={v['N']}")
    log.info("=" * 60)

    output = {
        "step": "step_183",
        "description": "JADES DR5 morphology × DR4 spec-z: compactness vs Gamma_t",
        "n_matched": int(len(matched_z)),
        "n_with_muv": int(valid_muv.sum()),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

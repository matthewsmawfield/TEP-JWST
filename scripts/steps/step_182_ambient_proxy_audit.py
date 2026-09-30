"""
Step 182: Ambient-Field Proxy Audit
====================================

Audits the u_amb proxy used by the step_176 covariance models
(Cov_uamb / Cov_Std_uamb / Cov_Int_uamb).

The step_176 ambient proxy is a k-nearest-neighbour density in
standardized (RA, Dec, log(1+z)) space: log u_amb ∝ -1.5 log d_k,
tracking the theory-0 ambient closure u_amb ∝ sqrt(rho_env * L).
Because the z-axis can dominate the neighbour metric, this step
decomposes the proxy's evidence gain into axis components and
z-orthogonalized variants, testing whether the gain is a genuine
sky/environment direction or a redshift-smoothing artefact:

  - uamb_3d    : kNN density in (RA, Dec, log1p z)  [the step_176 proxy]
  - uamb_sky   : kNN density in (RA, Dec) only
  - uamb_zonly : kNN density in log1p(z) only
  - uamb_perpz : uamb_3d linearly residualized against z

Each variant enters the covariance likelihood as {m, z, uamb_x}
(4K+1 params); the strongest surviving variant is additionally
evaluated on top of the free interaction, {m, z, m*z, uamb_x}
(5K+1 params).  Reference models {m,z} and {m,z,mz} are rerun for
a within-audit baseline.  A clustercentric-distance comparator,
d_A2744, is then run in the same slots ({m,z,d_cl} and
{m,z,mz,d_cl}) together with the sky proxy residualized on it.

The corrected environmental ladder then recognises that A2744 is a
FOREGROUND cluster (z = 0.308): cluster-centric distance cannot be
an ambient-field coordinate for z>8 galaxies — it is a systematics
coordinate (lensing-magnification bias and the mosaic-depth gradient
are both radially centred on the core).  The lens-model magnification
mu_ml is joined from the DR4 SPS catalogue, and the theory-faithful
ambient proxy is rebuilt as uamb_mw, the mass-weighted projected
density of the z>8 epoch slice (the photo-z widths, sigma_z ~ 4.8,
make all z>=8 sources one slice; u_amb ∝ sqrt(rho_env * L) uses the
mass density).  The decisive tests are the systematics-controlled
models {m,z,mu,d_cl,uamb_mw} and {m,z,mz,mu,d_cl,uamb_mw}, plus their
physical-subset (no chi2) versions.

Outputs: results/outputs/step_182_ambient_proxy_audit.json

Author: Matthew Lukin Smawfield
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.steps.step_176_nested_bayesian_evidence import (
    load_data, compute_ml_response_self_consistent, run_nested,
    OBSERVABLES, ALPHA_NUCLEAR,
    _joint_cov_standard_loglike, _joint_cov_standard_prior,
    _joint_cov_mz_interaction_loglike, _joint_cov_mz_interaction_prior,
    _joint_cov_mz_plus_rml_loglike, _joint_cov_mz_plus_rml_prior,
    _joint_cov_three_pred_loglike, _joint_cov_three_pred_prior,
)

STEP_NAME = "step_182_ambient_proxy_audit"


def _cov_npred_loglike(params, obs_matrix, preds, cov_inv, cov_logdet):
    """Generic joint-covariance likelihood with P predictors per observable:
    obs_k = a_k + sum_j b_jk * preds[:, j].  Total: (P+1)K + 1 params."""
    K = obs_matrix.shape[1]
    P = preds.shape[1]
    alpha = np.exp(params[(P + 1) * K])
    pred = np.empty_like(obs_matrix)
    for k in range(K):
        idx = (P + 1) * k
        pred[:, k] = params[idx] + preds @ params[idx + 1:idx + 1 + P]
    resid = obs_matrix - pred
    scaled = cov_inv / alpha
    logdet = cov_logdet + K * np.log(alpha)
    qf = np.einsum('ni,ij,nj->n', resid, scaled, resid)
    return -0.5 * (qf.sum() + resid.shape[0] * (K * np.log(2 * np.pi) + logdet))


def _cov_npred_prior(u, K, P):
    """Generic prior: intercepts U(-10,10), slopes U(-2,2), log_alpha U(-2,2)."""
    out = np.empty((P + 1) * K + 1)
    for k in range(K):
        idx = (P + 1) * k
        out[idx] = u[idx] * 20 - 10
        out[idx + 1:idx + 1 + P] = u[idx + 1:idx + 1 + P] * 4 - 2
    out[(P + 1) * K] = u[(P + 1) * K] * 4 - 2
    return out


def make_uamb_mw(df_v, sigma_arcmin):
    """Theory-faithful ambient proxy: log of the mass-weighted projected
    density of the same-epoch (z>=8) tracer field, Gaussian kernel with
    scale sigma (leave-one-out).  u_amb ∝ sqrt(rho_env * L); the proxy is
    log rho_M at kernel scale L = sigma."""
    ra = pd.to_numeric(df_v['ra'], errors='coerce').values
    dec = pd.to_numeric(df_v['dec'], errors='coerce').values
    x = (ra - np.nanmean(ra)) * np.cos(np.radians(np.nanmean(dec))) * 60.0
    y = (dec - np.nanmean(dec)) * 60.0
    w = 10.0 ** df_v['log_Mstar'].values
    w = np.where(np.isfinite(w), w, 0.0)
    cs = np.column_stack([x, y])
    tree = cKDTree(cs)
    n = len(cs)
    rho = np.zeros(n)
    for i in range(n):
        idx = tree.query_ball_point(cs[i], 3.0 * sigma_arcmin)
        d2 = np.sum((cs[idx] - cs[i]) ** 2, axis=1)
        rho[i] = np.sum(w[idx] * np.exp(-0.5 * d2 / sigma_arcmin ** 2))
        rho[i] -= w[i]  # leave-one-out
    rho = np.maximum(rho, 1e-10)
    lu = np.log(rho)
    return (lu - lu.mean()) / lu.std(ddof=0)


def load_lens_mu(df_v):
    """Join the UNCOVER DR4 v2.0 lensing-model magnification (mu_ml) onto
    the model sample by source id.  mu is a smooth radially centred
    systematic: near-core high-z sources are magnified (mu up to ~10)."""
    from astropy.io import fits
    cat = (PROJECT_ROOT / 'data' / 'raw' / 'uncover'
           / 'UNCOVER_DR4_SPS_catalog.fits')
    with fits.open(cat, memmap=True) as h:
        cid = np.asarray(h[1].data['id']).astype(int)
        mu = np.asarray(h[1].data['mu_ml'], dtype=float)
    mumap = dict(zip(cid, mu))
    muv = np.array([mumap.get(int(i), np.nan) for i in df_v['id']])
    n_miss = int(np.isnan(muv).sum())
    if n_miss:
        print_status(f"mu_ml join: {n_miss} unmatched ids -> mu=1", "WARNING")
        muv = np.where(np.isfinite(muv), muv, 1.0)
    return np.log10(muv)


def _lensing_control(df_v):
    """Does the headline dust–R_ML correlation survive magnification and
    clustercentric control?  The correlation must persist in the
    unmagnified subset and in both radial halves."""
    from scipy.stats import spearmanr
    lmu = load_lens_mu(df_v)
    mu = 10.0 ** lmu
    ra_c, dec_c = 3.5963, -30.3883
    dcl = np.hypot(
        (df_v['ra'].values - ra_c) * np.cos(np.radians(dec_c)),
        df_v['dec'].values - dec_c)
    g = df_v['gamma_t'].values
    du = df_v['dust'].values
    out = {}
    r, p = spearmanr(g, du)
    out['rho_full'] = [float(r), float(p)]
    lo = mu <= 1.05
    r, p = spearmanr(g[lo], du[lo])
    out['rho_mu_le_1p05'] = [float(r), float(p), int(lo.sum())]
    med = np.median(dcl)
    r, p = spearmanr(g[dcl > med], du[dcl > med])
    out['rho_dcl_outer_half'] = [float(r), float(p), int((dcl > med).sum())]
    r, p = spearmanr(g[dcl <= med], du[dcl <= med])
    out['rho_dcl_inner_half'] = [float(r), float(p), int((dcl <= med).sum())]
    r, p = spearmanr(mu, du)
    out['rho_mu_dust'] = [float(r), float(p)]
    r, p = spearmanr(dcl, du)
    out['rho_dcl_dust'] = [float(r), float(p)]
    # Magnification-corrected-mass variant: if catalog mstar_50 is not
    # already mu-corrected, high-mu sources have inflated masses ->
    # inflated R_ML at the core.  Recompute the kernel on
    # log_Mstar_corr = log_Mstar - log10(mu_ml) and re-test.
    try:
        mass_v = df_v['log_Mstar'].values - lmu
        z_v = df_v['z'].values
        g_corr, _ = compute_ml_response_self_consistent(
            mass_v, z_v, n_sps=ALPHA_NUCLEAR)
        ok = np.isfinite(g_corr)
        r, p = spearmanr(g_corr[ok], du[ok])
        out['rho_mu_corrected_mass'] = [float(r), float(p), int(ok.sum())]
        r, p = spearmanr(g_corr[ok][lo[ok]], du[ok][lo[ok]])
        out['rho_mu_corr_mu_le_1p05'] = [float(r), float(p),
                                       int((ok & lo).sum())]
    except Exception as e:
        out['rho_mu_corrected_mass'] = f'failed: {e}'
    # Photometry-quality sensitivity: the step_001 cuts retain
    # use_phot != 1 sources; the headline must survive the survey's
    # own photometry flag (the DR4 path in step_152 applies it).
    if 'use_phot' in df_v.columns:
        up = df_v['use_phot'].values == 1
        out['n_use_phot_flagged'] = int((~up).sum())
        r, p = spearmanr(g[up], du[up])
        out['rho_use_phot_1'] = [float(r), float(p), int(up.sum())]
        loq = lo & up
        r, p = spearmanr(g[loq], du[loq])
        out['rho_mu_le_1p05_use_phot_1'] = [float(r), float(p),
                                          int(loq.sum())]
    return out


def make_uamb(df_v, z, axes):
    """kNN ambient proxy: log u ∝ -1.5 log d_5 in standardized axes."""
    vals = {
        'ra': pd.to_numeric(df_v['ra'], errors='coerce').values,
        'dec': pd.to_numeric(df_v['dec'], errors='coerce').values,
        'z': np.log1p(z),
    }
    cols = []
    for a in axes:
        v = vals[a]
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        cols.append((v - v.mean()) / v.std(ddof=0))
    cs = np.column_stack(cols)
    d = np.maximum(cKDTree(cs).query(cs, k=6)[0][:, 5], 1e-6)
    lu = -1.5 * np.log(d)
    return (lu - lu.mean()) / lu.std(ddof=0)


def main():
    logger = TEPLogger(STEP_NAME,
                       log_file_path=PROJECT_ROOT / "logs" / f"{STEP_NAME}.log")
    set_step_logger(logger)
    print_status(f"STEP 182: Ambient-field proxy audit", "TITLE")

    df = load_data()
    rc = 'ml_response' if 'ml_response' in df.columns else 'gamma_t'
    cols_needed = ['dust', 'log_Mstar', 'z', rc] + [
        c for c in OBSERVABLES if c != 'dust']
    df_v = df[df[cols_needed].notna().all(axis=1)].copy()

    mass = df_v['log_Mstar'].values
    z = df_v['z'].values
    obs_raw = [df_v[c].values for c in OBSERVABLES]
    obs = [(a - a.mean()) / a.std(ddof=0) for a in obs_raw]
    obs_matrix = np.column_stack(obs)
    K = len(OBSERVABLES)
    cov = np.cov(obs_matrix, rowvar=False) + 1e-4 * np.eye(K)
    cov_inv = np.linalg.inv(cov)
    cov_logdet = float(np.linalg.slogdet(cov)[1])
    mass_std = (mass - mass.mean()) / mass.std(ddof=0)
    z_std = (z - z.mean()) / z.std(ddof=0)

    gamma_t, _ = compute_ml_response_self_consistent(
        mass, z, n_sps=ALPHA_NUCLEAR)
    lg = np.log10(np.maximum(gamma_t, 0.01))
    lg = (lg - lg.mean()) / lg.std(ddof=0)

    # Proxy variants
    uamb = {
        'uamb_3d': make_uamb(df_v, z, ['ra', 'dec', 'z']),
        'uamb_sky': make_uamb(df_v, z, ['ra', 'dec']),
        'uamb_zonly': make_uamb(df_v, z, ['z']),
    }
    A = np.column_stack([np.ones_like(z_std), z_std])
    beta = np.linalg.lstsq(A, uamb['uamb_3d'], rcond=None)[0]
    uamb['uamb_perpz'] = uamb['uamb_3d'] - A @ beta
    uamb['uamb_perpz'] = (uamb['uamb_perpz'] - uamb['uamb_perpz'].mean()) \
        / uamb['uamb_perpz'].std(ddof=0)

    # Lensing magnification (v2.0 lens model, mu_ml) and mass-weighted
    # same-epoch density.  A2744 is a FOREGROUND cluster (z=0.308): it
    # cannot be part of a z>8 galaxy's ambient field, so d_A2744 and mu
    # are systematics coordinates (lensing magnification + mosaic depth
    # gradient), not ambient-field proxies.  The theory-faithful ambient
    # proxy is the projected density of the z>8 population itself (the
    # sigma_z ~ 4.8 photo-z widths make all z>=8 sources one epoch
    # slice), mass-weighted because u_amb ∝ sqrt(rho_env * L) uses the
    # mass density.  The decisive test is whether it adds evidence on
    # top of the systematics baseline {m,z,mu,d_cl}.
    lmu = load_lens_mu(df_v)
    lmu_s = (lmu - lmu.mean()) / np.maximum(lmu.std(ddof=0), 1e-12)
    uamb['uamb_mw_1arcmin'] = make_uamb_mw(df_v, 1.0)   # ~3.6 cMpc at z~9
    uamb['uamb_mw_03arcmin'] = make_uamb_mw(df_v, 0.3)  # ~1.1 cMpc at z~9
    uamb_mw = uamb['uamb_mw_1arcmin']

    corr_table = {
        nm: {
            'corr_mass': float(np.corrcoef(u, mass_std)[0, 1]),
            'corr_z': float(np.corrcoef(u, z_std)[0, 1]),
            'corr_log_RML': float(np.corrcoef(u, lg)[0, 1]),
        }
        for nm, u in uamb.items()
    }
    for nm, c in corr_table.items():
        print_status(
            f"{nm}: corr(m)={c['corr_mass']:+.2f} "
            f"corr(z)={c['corr_z']:+.2f} corr(lg R_ML)={c['corr_log_RML']:+.2f}",
            "INFO")

    results = {'proxy_correlations': corr_table, 'models': {}}

    def rec(key, ll, pr, ndim, label):
        r = run_nested(ll, pr, ndim, label)
        results['models'][key] = {
            'logZ': float(r['logZ']), 'logZ_err': float(r['logZ_err']),
            'n_params': int(ndim)}
        return r['logZ']

    # within-audit baselines
    lz_mz0 = rec('m_z',
                 lambda p: _joint_cov_standard_loglike(
                     p, obs_matrix, mass_std, z_std, cov_inv, cov_logdet),
                 lambda u: _joint_cov_standard_prior(u, K),
                 3 * K + 1, "{m,z}")
    lz_int = rec('m_z_mz',
                 lambda p: _joint_cov_mz_interaction_loglike(
                     p, obs_matrix, mass_std, z_std, cov_inv, cov_logdet),
                 lambda u: _joint_cov_mz_interaction_prior(u, K),
                 4 * K + 1, "{m,z,mz}")

    for nm, u in uamb.items():
        rec(f'm_z_{nm}',
            lambda p, uu=u: _joint_cov_three_pred_loglike(
                p, obs_matrix, mass_std, z_std, uu, cov_inv, cov_logdet),
            lambda u: _joint_cov_three_pred_prior(u, K),
            4 * K + 1, f"{{m,z,{nm}}}")

    # strongest variant on top of the free interaction (5K+1)
    rec('m_z_mz_uamb_sky',
        lambda p: _joint_cov_mz_plus_rml_loglike(
            p, obs_matrix, mass_std, z_std, uamb['uamb_sky'],
            cov_inv, cov_logdet),
        lambda u: _joint_cov_mz_plus_rml_prior(u, K),
        5 * K + 1, "{m,z,mz,uamb_sky}")

    # Clustercentric-distance comparator: is the sky proxy anything beyond
    # the radial coordinate about the A2744 core?  If d_cl alone matches or
    # beats the kNN proxy, and the proxy residualized on d_cl retains no
    # evidence, the environmental gain is carried by the smooth radial
    # coordinate -- geometrically what an ambient-depth field in the cluster
    # well predicts, and also what a mosaic depth/selection gradient centred
    # on the same point predicts.  The two readings are not separable here.
    ra_c, dec_c = 3.5963, -30.3883
    dcl = np.hypot((df_v['ra'].values - ra_c) * np.cos(np.radians(dec_c)),
                   df_v['dec'].values - dec_c)
    dcl_s = (dcl - dcl.mean()) / dcl.std(ddof=0)
    Ad = np.column_stack([np.ones_like(dcl_s), dcl_s])
    u_sky_perp_dcl = uamb['uamb_sky'] - Ad @ np.linalg.lstsq(
        Ad, uamb['uamb_sky'], rcond=None)[0]
    u_sky_perp_dcl = ((u_sky_perp_dcl - u_sky_perp_dcl.mean())
                      / u_sky_perp_dcl.std(ddof=0))

    rec('m_z_d_A2744',
        lambda p: _joint_cov_three_pred_loglike(
            p, obs_matrix, mass_std, z_std, dcl_s, cov_inv, cov_logdet),
        lambda u: _joint_cov_three_pred_prior(u, K),
        4 * K + 1, "{m,z,d_A2744}")
    rec('m_z_uamb_sky_perp_dcl',
        lambda p: _joint_cov_three_pred_loglike(
            p, obs_matrix, mass_std, z_std, u_sky_perp_dcl,
            cov_inv, cov_logdet),
        lambda u: _joint_cov_three_pred_prior(u, K),
        4 * K + 1, "{m,z,uamb_sky_perp_dcl}")
    rec('m_z_mz_d_A2744',
        lambda p: _joint_cov_mz_plus_rml_loglike(
            p, obs_matrix, mass_std, z_std, dcl_s, cov_inv, cov_logdet),
        lambda u: _joint_cov_mz_plus_rml_prior(u, K),
        5 * K + 1, "{m,z,mz,d_A2744}")

    # ------------------------------------------------------------------
    # Lensing/systematics-controlled ladder.  mu_ml is the lens-model
    # magnification; dcl_s is the cluster-centric coordinate.  Because the
    # foreground cluster cannot belong to the z>8 ambient field, any
    # environmental signal must add evidence on top of {m,z,mu,d_cl} —
    # this is the corrected version of the earlier proxy-vs-dcl framing.
    def recn(key, pred_list, label):
        P = len(pred_list)
        preds = np.column_stack(pred_list)
        r = run_nested(
            lambda p, pr=preds: _cov_npred_loglike(
                p, obs_matrix, pr, cov_inv, cov_logdet),
            lambda u, PP=P: _cov_npred_prior(u, K, PP),
            (P + 1) * K + 1, label)
        results['models'][key] = {
            'logZ': float(r['logZ']), 'logZ_err': float(r['logZ_err']),
            'n_params': int((P + 1) * K + 1)}
        return r['logZ']

    recn('m_z_mu', [mass_std, z_std, lmu_s], "{m,z,mu}")
    recn('m_z_dcl_mu', [mass_std, z_std, dcl_s, lmu_s],
         "{m,z,d_A2744,mu}")
    recn('m_z_dcl_mu_uamb_mw', [mass_std, z_std, dcl_s, lmu_s, uamb_mw],
         "{m,z,d_A2744,mu,uamb_mw}")
    recn('m_z_mz_dcl_mu_uamb_mw',
         [mass_std, z_std, mass_std * z_std, dcl_s, lmu_s, uamb_mw],
         "{m,z,mz,d_A2744,mu,uamb_mw}")
    recn('m_z_uamb_sky_on_dcl_mu',
         [mass_std, z_std, dcl_s, lmu_s, uamb['uamb_sky']],
         "{m,z,d_A2744,mu,uamb_sky}")

    # Physical-subset control (dust, sSFR, metallicity — chi2 dropped).
    # The chi2 channel is a fit-quality diagnostic and tracks the mosaic
    # depth gradient, so this isolates whether the proxy gain survives
    # on physical observables alone.
    OBS3 = ['dust', 'log_ssfr', 'met']
    obs3 = np.column_stack([
        (df_v[c].values - df_v[c].mean()) / df_v[c].std(ddof=0)
        for c in OBS3])
    K3 = len(OBS3)
    cov3 = np.cov(obs3, rowvar=False) + 1e-4 * np.eye(K3)
    ci3 = np.linalg.inv(cov3)
    cld3 = float(np.linalg.slogdet(cov3)[1])

    def rec3(key, uamb_arr, label):
        if uamb_arr is None:
            r = run_nested(
                lambda p: _joint_cov_standard_loglike(
                    p, obs3, mass_std, z_std, ci3, cld3),
                lambda u: _joint_cov_standard_prior(u, K3),
                3 * K3 + 1, label)
        else:
            r = run_nested(
                lambda p: _joint_cov_three_pred_loglike(
                    p, obs3, mass_std, z_std, uamb_arr, ci3, cld3),
                lambda u: _joint_cov_three_pred_prior(u, K3),
                4 * K3 + 1, label)
        results['models'][key] = {'logZ': float(r['logZ']),
                                  'logZ_err': float(r['logZ_err']),
                                  'n_params': int(r.get('n_params', 0) or 0)}
        return r['logZ']

    lz3_mz = rec3('m_z_phys3', None, "{m,z} phys-subset")
    lz3_u = rec3('m_z_uamb_sky_phys3', uamb['uamb_sky'],
                 "{m,z,uamb_sky} phys-subset")
    results['delta_uamb_sky_over_mz_physical_subset'] = lz3_u - lz3_mz

    # Physical-subset versions of the systematics-controlled ladder:
    # {m,z,mu,dcl} baseline and {m,z,mu,dcl,uamb_mw} on physical
    # observables only — the cleanest possible ambient-field test.
    def rec3n(key, pred_list, label):
        P = len(pred_list)
        preds = np.column_stack(pred_list)
        r = run_nested(
            lambda p, pr=preds: _cov_npred_loglike(
                p, obs3, pr, ci3, cld3),
            lambda u, PP=P: _cov_npred_prior(u, K3, PP),
            (P + 1) * K3 + 1, label)
        results['models'][key] = {'logZ': float(r['logZ']),
                                  'logZ_err': float(r['logZ_err']),
                                  'n_params': int((P + 1) * K3 + 1)}
        return r['logZ']

    rec3n('m_z_uamb_mw_phys3', [mass_std, z_std, uamb_mw],
          "{m,z,uamb_mw} phys-subset")
    rec3n('m_z_dcl_mu_phys3', [mass_std, z_std, dcl_s, lmu_s],
          "{m,z,d_A2744,mu} phys-subset")
    rec3n('m_z_dcl_mu_uamb_mw_phys3',
          [mass_std, z_std, dcl_s, lmu_s, uamb_mw],
          "{m,z,d_A2744,mu,uamb_mw} phys-subset")

    # Cluster-centric diagnostic: how much of the sky proxy is just
    # distance from the Abell-2744 core (mosaic depth tracks it too).
    results['cluster_centric_diagnostic'] = {
        'corr_uamb_sky_dist_A2744': float(np.corrcoef(uamb['uamb_sky'], dcl)[0, 1]),
        'corr_uamb_mw_dist_A2744': float(np.corrcoef(uamb_mw, dcl)[0, 1]),
        'corr_uamb_mw_uamb_sky': float(np.corrcoef(uamb_mw, uamb['uamb_sky'])[0, 1]),
        'corr_mu_dist_A2744': float(np.corrcoef(lmu_s, dcl)[0, 1]),
        'corr_mu_mass': float(np.corrcoef(lmu_s, mass_std)[0, 1]),
        'corr_mu_chi2': float(np.corrcoef(
            lmu_s, (df_v['chi2'].values - df_v['chi2'].mean())
            / df_v['chi2'].std(ddof=0))[0, 1]),
        'mu_ml_stats': {
            'median': float(np.median(10.0 ** lmu)),
            'max': float(np.max(10.0 ** lmu)),
            'n_above_1p5': int((10.0 ** lmu > 1.5).sum()),
        },
        'headline_correlation_lensing_control': _lensing_control(df_v),
        'note': ("A2744 is a foreground cluster (z=0.308): it cannot be "
                 "part of a z>8 galaxy's ambient field, so d_A2744 and "
                 "mu_ml are systematics coordinates (lensing magnification "
                 "bias and mosaic-depth gradient), not ambient proxies. "
                 "The theory-faithful ambient proxy is the mass-weighted "
                 "projected density of the z>8 slice itself (uamb_mw): "
                 "u_amb ∝ sqrt(rho_env * L) uses the same-epoch mass "
                 "density. The decisive ambient-field test is whether "
                 "uamb_mw adds evidence on top of {m,z,mu,d_A2744}; the "
                 "physical-subset versions drop chi2 to isolate physical "
                 "observables from the fit-quality/depth channel."),
    }

    results['delta_logZ_vs_m_z'] = {
        k: v['logZ'] - lz_mz0 for k, v in results['models'].items()
        if k != 'm_z'}
    results['delta_uamb_sky_over_interaction'] = (
        results['models']['m_z_mz_uamb_sky']['logZ'] - lz_int)

    out = PROJECT_ROOT / "results" / "outputs" / f"{STEP_NAME}.json"
    import json
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"Results saved to {out}", "SUCCESS")


if __name__ == "__main__":
    main()

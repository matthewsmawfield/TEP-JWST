import numpy as np
from scipy import stats


def _as_2d_controls(controls):
    if isinstance(controls, np.ndarray):
        arr = np.asarray(controls, dtype=float)
        if arr.ndim == 1:
            return arr[:, None]
        return arr
    cols = [np.asarray(c, dtype=float) for c in controls]
    if not cols:
        raise ValueError("At least one control variable is required.")
    return np.column_stack(cols)


def rankdata_with_nan(x):
    arr = np.asarray(x, dtype=float)
    ranks = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.any():
        ranks[mask] = stats.rankdata(arr[mask])
    return ranks


def residualize(values, controls):
    vals = np.asarray(values, dtype=float)
    ctrl = _as_2d_controls(controls)
    resid = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals) & np.all(np.isfinite(ctrl), axis=1)
    if mask.sum() < 4:
        return resid, mask
    xmat = np.column_stack([ctrl[mask], np.ones(mask.sum(), dtype=float)])
    beta = np.linalg.lstsq(xmat, vals[mask], rcond=None)[0]
    resid[mask] = vals[mask] - xmat @ beta
    return resid, mask


def partial_rank_correlation(x, y, controls):
    ctrl = _as_2d_controls(controls)
    x_rank = rankdata_with_nan(x)
    y_rank = rankdata_with_nan(y)
    ctrl_rank = np.column_stack([rankdata_with_nan(ctrl[:, i]) for i in range(ctrl.shape[1])])
    x_resid, mask_x = residualize(x_rank, ctrl_rank)
    y_resid, mask_y = residualize(y_rank, ctrl_rank)
    mask = mask_x & mask_y & np.isfinite(x_resid) & np.isfinite(y_resid)
    if mask.sum() < 4:
        return 0.0, 1.0, int(mask.sum())
    rho, p = stats.pearsonr(x_resid[mask], y_resid[mask])
    return float(rho), float(p), int(mask.sum())


def bootstrap_partial_rank_ci(x, y, controls, n_boot=1000, seed=42):
    ctrl = _as_2d_controls(controls)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(ctrl), axis=1)
    x = x[mask]
    y = y[mask]
    ctrl = ctrl[mask]
    n = len(x)
    if n < 4:
        return np.array([np.nan, np.nan], dtype=float)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        rho, _, _ = partial_rank_correlation(x[idx], y[idx], ctrl[idx])
        if np.isfinite(rho):
            rhos.append(float(rho))
    if len(rhos) < 30:
        return np.array([np.nan, np.nan], dtype=float)
    return np.percentile(rhos, [2.5, 97.5]).astype(float)

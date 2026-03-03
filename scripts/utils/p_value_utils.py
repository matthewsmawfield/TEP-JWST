#!/usr/bin/env python3
"""
P-value utilities for TEP-JWST pipeline.

Provides robust p-value handling to avoid:
- Exact 0.0 values (causes infinite significance interpretation)
- Infinity in log-space calculations
- JSON serialization issues
"""

import numpy as np
from scipy import stats

# Minimum representable p-value (avoids 0.0)
MIN_P_VALUE = 1e-300

# Minimum log10(p) to avoid -Infinity
MIN_LOG10_P = -308.0


def format_p_value(p, min_p=MIN_P_VALUE):
    """
    Format p-value for JSON serialization, ensuring no exact 0.0 values.
    
    Parameters
    ----------
    p : float
        Raw p-value from statistical test
    min_p : float
        Minimum p-value floor (default: 1e-300)
    
    Returns
    -------
    float
        P-value with minimum floor applied
    """
    if p is None:
        return None
    if np.isnan(p):
        return None
    if p == 0.0 or (p < min_p and p >= 0):
        return float(min_p)
    return float(p)


def safe_json_default(obj):
    """
    JSON serializer for numpy/non-standard types.
    
    Use as: json.dump(data, f, indent=2, default=safe_json_default)
    
    Converts numpy types to native Python types instead of
    stringifying them (which would bypass p-value guardrails).
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


def p_value_from_t(t_stat, df, two_tailed=True):
    """
    Compute p-value from t-statistic with robust handling.
    
    Parameters
    ----------
    t_stat : float
        T-statistic value
    df : float
        Degrees of freedom
    two_tailed : bool
        Whether to compute two-tailed p-value
    
    Returns
    -------
    dict
        Dictionary with 'p_value', 'log10_p', 'p_formatted'
    """
    log_sf = float(stats.t.logsf(abs(t_stat), df))
    if two_tailed:
        log_p = float(np.log(2.0) + log_sf)
    else:
        log_p = log_sf
    
    # Compute p-value if representable
    min_log = float(np.log(MIN_P_VALUE))
    p_val = None
    if log_p >= min_log:
        p_val = float(np.exp(log_p))
    
    # Compute log10(p) with floor
    log10_p = float(log_p / np.log(10.0))
    if log10_p < MIN_LOG10_P or np.isinf(log10_p):
        log10_p = MIN_LOG10_P
    
    # Format string
    if p_val is not None:
        if p_val >= 1e-3:
            p_formatted = f"{p_val:.3g}"
        else:
            p_formatted = f"{p_val:.2e}"
    else:
        p_formatted = f"10^({log10_p:.1f})"
    
    return {
        'p_value': format_p_value(p_val),
        'log10_p': log10_p,
        'p_formatted': p_formatted,
    }


def spearman_with_p(x, y):
    """
    Compute Spearman correlation with robust p-value.
    
    Parameters
    ----------
    x, y : array-like
        Data arrays
    
    Returns
    -------
    dict
        Dictionary with 'rho', 'p_value', 'n'
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    n = len(x)
    if n < 3:
        return {'rho': None, 'p_value': None, 'n': n}
    
    rho, p = stats.spearmanr(x, y)
    
    return {
        'rho': float(rho),
        'p_value': format_p_value(p),
        'n': n,
    }

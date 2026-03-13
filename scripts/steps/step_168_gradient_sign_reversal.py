import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import safe_json_default  # JSON serialiser for numpy types

STEP_NUM = "168"  # Pipeline step number
STEP_NAME = "gradient_sign_reversal"  # Used in log / output filenames
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
STEP117_JSON = OUTPUT_PATH / "step_117_dynamical_mass_comparison.json"  # L4 dynamical mass results
STEP159_JSON = OUTPUT_PATH / "step_159_mass_measurement_bias.json"  # Mass bias calibration
for path in [OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

BOOTSTRAP_N = 4000
RNG_SEED = 168


def _clip_p(value: float) -> float:
    return max(float(value), 1e-300)


def _load_beta_debias() -> tuple[float, str]:
    if STEP117_JSON.exists():
        try:
            s117 = json.loads(STEP117_JSON.read_text())
            beta_boot = s117.get("object_level_beta_bootstrap")
            if isinstance(beta_boot, dict) and beta_boot.get("median") is not None:
                return float(beta_boot["median"]), "step_117_object_level_bootstrap"
        except Exception:
            pass
    if STEP159_JSON.exists():
        try:
            s159 = json.loads(STEP159_JSON.read_text())
            beta_emp = s159.get("beta_empirical_from_L4")
            if beta_emp is not None and np.isfinite(beta_emp):
                return float(beta_emp), "step_159_l4_empirical"
        except Exception:
            pass
    return 0.7, "fallback_beta_ml_prior"


def _load_resolved_with_tep() -> pd.DataFrame:
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "steps" / "step_139_colour_gradient_steiger.py"
    spec = importlib.util.spec_from_file_location("step139", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.load_jades_with_tep()
    if df is None:
        raise FileNotFoundError("Resolved JADES gradient input could not be loaded via step_139.")
    z_col = "z_phot" if "z_phot" in df.columns else "z"
    df = df.dropna(subset=["colour_gradient", "gamma_t", "log_mstar", z_col]).copy()
    df = df.rename(columns={z_col: "z"})
    return df


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(BOOTSTRAP_N, dtype=float)
    for i in range(BOOTSTRAP_N):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        out[i] = float(np.mean(a[ia]) - np.mean(b[ib]))
    return out


def _residualize(values: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones(len(values))] + [np.asarray(c, dtype=float) for c in controls])
    beta = np.linalg.lstsq(X, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - X @ beta


def _sign_table(high: pd.DataFrame, low: pd.DataFrame, col: str) -> dict:
    a = int((high[col] < 0).sum())
    b = int((high[col] >= 0).sum())
    c = int((low[col] < 0).sum())
    d = int((low[col] >= 0).sum())
    odds_ratio, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "negative_fraction_high": float(a / len(high)),
        "negative_fraction_low": float(c / len(low)),
        "contingency_table": [[a, b], [c, d]],
        "odds_ratio_high_more_negative": None if np.isnan(odds_ratio) else float(odds_ratio),
        "fisher_p_one_sided": _clip_p(p),
    }


def _group_contrast(high: pd.DataFrame, low: pd.DataFrame, gradient_col: str, label: str, metadata: dict | None = None, min_group_size: int = 20) -> dict | None:
    if len(low) < min_group_size or len(high) < min_group_size:
        return None
    rng = np.random.default_rng(RNG_SEED + sum(ord(ch) for ch in label))
    boot = _bootstrap_delta(high[gradient_col].to_numpy(dtype=float), low[gradient_col].to_numpy(dtype=float), rng)
    mw = mannwhitneyu(high[gradient_col], low[gradient_col], alternative="two-sided")
    summary = _sign_table(high, low, gradient_col)
    summary.update({
        "contrast_label": label,
        "median_gradient_high": float(np.nanmedian(high[gradient_col])),
        "median_gradient_low": float(np.nanmedian(low[gradient_col])),
        "mean_gradient_high": float(np.nanmean(high[gradient_col])),
        "mean_gradient_low": float(np.nanmean(low[gradient_col])),
        "delta_mean_high_minus_low": float(np.nanmean(high[gradient_col]) - np.nanmean(low[gradient_col])),
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_one_sided_p_high_less_than_low": _clip_p((float(np.sum(boot >= 0.0)) + 1.0) / (len(boot) + 1.0)),
        "mannwhitney_p_two_sided": _clip_p(mw.pvalue),
    })
    if metadata:
        summary.update(metadata)
    return summary


def _quantile_contrast(df: pd.DataFrame, gradient_col: str, low_q: float, high_q: float) -> dict | None:
    low_thr = float(df["gamma_t"].quantile(low_q))
    high_thr = float(df["gamma_t"].quantile(high_q))
    low = df[df["gamma_t"] <= low_thr].copy()
    high = df[df["gamma_t"] >= high_thr].copy()
    summary = _group_contrast(
        high,
        low,
        gradient_col,
        f"quantile_{low_q:.3f}_{high_q:.3f}_{gradient_col}",
        {
        "gamma_t_quantiles": [float(low_q), float(high_q)],
        "gamma_t_threshold_low": low_thr,
        "gamma_t_threshold_high": high_thr,
        },
    )
    return summary


def _assessment(primary: dict | None, literal_gamma_gt_1_available: bool) -> str:
    if not literal_gamma_gt_1_available and primary is None:
        return "The current resolved sample does not support a literal Gamma_t>1 versus Gamma_t<1 inversion test and no stable high-vs-low screening contrast could be formed."
    if primary is None:
        return "No stable resolved sign-reversal contrast could be computed from the current JADES sample."
    if primary["negative_fraction_high"] > primary["negative_fraction_low"] and primary["bootstrap_ci_95"][1] < 0:
        return "The resolved sample shows the TEP-sign gradient reversal most clearly in the high-screening tail: high-Gamma_t galaxies are more likely to show the negative gradient convention associated with bluer cores."
    if primary["negative_fraction_high"] > primary["negative_fraction_low"]:
        return "The resolved sign test is directionally consistent with TEP after converting the problem into a high-vs-low screening comparison, but the effect remains short of a decisive inversion detection."
    return "The resolved sign test does not currently recover a clean TEP-style gradient inversion with existing JADES photometry."


def _select_primary_contrast(
    resid_literal: dict | None,
    resid_q33_debiased: dict | None,
    resid_q25_debiased: dict | None,
    resid_q33_observed: dict | None,
    resid_q25_observed: dict | None,
) -> tuple[dict | None, str]:
    if resid_literal is not None and min(resid_literal.get("n_high", 0), resid_literal.get("n_low", 0)) >= 20:
        return resid_literal, "literal_gt1_lt1_debiased_or_observed"
    if resid_q33_debiased is not None:
        return resid_q33_debiased, "quantile_33_67_debiased_mass_control"
    if resid_q25_debiased is not None:
        return resid_q25_debiased, "quantile_25_75_debiased_mass_control"
    if resid_literal is not None:
        return resid_literal, "literal_gt1_lt1_underpowered_but_only_available"
    if resid_q33_observed is not None:
        return resid_q33_observed, "quantile_33_67_observed_mass_control"
    return resid_q25_observed, "quantile_25_75_observed_mass_control"


def run():
    print_status(f"STEP {STEP_NUM}: Resolved gradient sign-reversal test", "INFO")
    beta_debias, beta_debias_source = _load_beta_debias()
    try:
        df = _load_resolved_with_tep()
    except FileNotFoundError as exc:
        result = {
            "step": STEP_NUM,
            "name": STEP_NAME,
            "status": "FAILED_NO_DATA",
            "note": str(exc),
        }
        (OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json").write_text(json.dumps(result, indent=2))
        return result

    literal_gamma_lt1 = int((df["gamma_t"] < 1).sum())
    literal_gamma_gt1 = int((df["gamma_t"] > 1).sum())
    df["log_gamma_t"] = np.log10(np.clip(df["gamma_t"].to_numpy(dtype=float), 1e-9, None))
    df["log_mstar_debiased"] = (
        df["log_mstar"].to_numpy(dtype=float) - beta_debias * df["log_gamma_t"].to_numpy(dtype=float)
    )

    df["gradient_resid_mass_z"] = _residualize(
        df["colour_gradient"].to_numpy(dtype=float),
        [df["log_mstar"].to_numpy(dtype=float), df["z"].to_numpy(dtype=float)],
    )
    df["gradient_resid_debiased_mass_z"] = _residualize(
        df["colour_gradient"].to_numpy(dtype=float),
        [df["log_mstar_debiased"].to_numpy(dtype=float), df["z"].to_numpy(dtype=float)],
    )

    raw_literal = None
    resid_literal = None
    if literal_gamma_gt1 > 0 and literal_gamma_lt1 > 0:
        low_literal = df[df["gamma_t"] < 1].copy()
        high_literal = df[df["gamma_t"] > 1].copy()
        raw_literal = _group_contrast(
            high_literal,
            low_literal,
            "colour_gradient",
            "literal_gamma_gt1_lt1_raw",
            {
                "gamma_t_threshold_low": 1.0,
                "gamma_t_threshold_high": 1.0,
                "literal_gamma_gt1_vs_lt1": True,
            },
            min_group_size=8,
        )
        resid_literal = _group_contrast(
            high_literal,
            low_literal,
            "gradient_resid_mass_z",
            "literal_gamma_gt1_lt1_resid",
            {
                "gamma_t_threshold_low": 1.0,
                "gamma_t_threshold_high": 1.0,
                "literal_gamma_gt1_vs_lt1": True,
            },
            min_group_size=8,
        )

    raw_q25 = _quantile_contrast(df, "colour_gradient", 0.25, 0.75)
    raw_q33 = _quantile_contrast(df, "colour_gradient", 1.0 / 3.0, 2.0 / 3.0)
    resid_q25 = _quantile_contrast(df, "gradient_resid_mass_z", 0.25, 0.75)
    resid_q33 = _quantile_contrast(df, "gradient_resid_mass_z", 1.0 / 3.0, 2.0 / 3.0)
    debiased_resid_literal = None
    if literal_gamma_gt1 > 0 and literal_gamma_lt1 > 0:
        low_literal = df[df["gamma_t"] < 1].copy()
        high_literal = df[df["gamma_t"] > 1].copy()
        debiased_resid_literal = _group_contrast(
            high_literal,
            low_literal,
            "gradient_resid_debiased_mass_z",
            "literal_gamma_gt1_lt1_resid_debiased_mass",
            {
                "gamma_t_threshold_low": 1.0,
                "gamma_t_threshold_high": 1.0,
                "literal_gamma_gt1_vs_lt1": True,
                "mass_control": "debiased",
            },
            min_group_size=8,
        )
    debiased_resid_q25 = _quantile_contrast(df, "gradient_resid_debiased_mass_z", 0.25, 0.75)
    debiased_resid_q33 = _quantile_contrast(df, "gradient_resid_debiased_mass_z", 1.0 / 3.0, 2.0 / 3.0)
    if resid_literal is not None:
        resid_literal["mass_control"] = "observed"
    if resid_q25 is not None:
        resid_q25["mass_control"] = "observed"
    if resid_q33 is not None:
        resid_q33["mass_control"] = "observed"
    if debiased_resid_literal is not None:
        debiased_resid_literal["mass_control"] = "debiased"
    if debiased_resid_q25 is not None:
        debiased_resid_q25["mass_control"] = "debiased"
    if debiased_resid_q33 is not None:
        debiased_resid_q33["mass_control"] = "debiased"
    primary, primary_selection = _select_primary_contrast(
        debiased_resid_literal,
        debiased_resid_q33,
        debiased_resid_q25,
        resid_q33,
        resid_q25,
    )

    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "Direct sign test for resolved JADES colour gradients across screening regimes",
        "prediction_table": {
            "standard_physics": "Standard inside-out growth predicts the standard-sign gradient to dominate across regimes.",
            "tep": "The high-screening tail should shift toward the TEP-sign gradient (negative colour_gradient in the current pipeline convention).",
            "pipeline_sign_convention": {
                "negative_colour_gradient": "TEP-sign / bluer-core direction in the current JADES gradient convention",
                "positive_colour_gradient": "standard-sign / redder-core direction in the current JADES gradient convention",
            },
        },
        "sample": {
            "n_total": int(len(df)),
            "gamma_t_min": float(np.nanmin(df["gamma_t"])),
            "gamma_t_median": float(np.nanmedian(df["gamma_t"])),
            "gamma_t_max": float(np.nanmax(df["gamma_t"])),
            "beta_debias_used": float(beta_debias),
            "beta_debias_source": beta_debias_source,
            "n_gamma_t_lt_1": literal_gamma_lt1,
            "n_gamma_t_gt_1": literal_gamma_gt1,
            "literal_gamma_gt1_vs_lt1_available": bool(literal_gamma_gt1 > 0 and literal_gamma_lt1 > 0),
            "note": (
                "The corrected step_139 mapping now uses direct halo masses from the local JADES physical catalog when available. "
                "That yields a small literal Gamma_t > 1 tail, but the sample remains dominated by Gamma_t < 1 objects, "
                "so both literal and quantile screening contrasts are reported."
            ),
        },
        "contrasts": {
            "raw_q25_q75": raw_q25,
            "raw_q33_q67": raw_q33,
            "raw_literal_gt1_lt1": raw_literal,
            "residual_q25_q75": resid_q25,
            "residual_q33_q67": resid_q33,
            "residual_literal_gt1_lt1": resid_literal,
            "residual_debiased_q25_q75": debiased_resid_q25,
            "residual_debiased_q33_q67": debiased_resid_q33,
            "residual_debiased_literal_gt1_lt1": debiased_resid_literal,
        },
        "primary_contrast": primary,
        "primary_selection": primary_selection,
    }
    result["assessment"] = _assessment(primary, bool(literal_gamma_gt1 > 0 and literal_gamma_lt1 > 0))

    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    out_json.write_text(json.dumps(result, indent=2, default=safe_json_default))
    if primary is not None:
        print_status(
            f"Primary residual gradient contrast Δ={primary['delta_mean_high_minus_low']:.3f} with 95% CI [{primary['bootstrap_ci_95'][0]:.3f}, {primary['bootstrap_ci_95'][1]:.3f}]",
            "INFO",
        )
    print_status(result["assessment"], "INFO")
    return result


if __name__ == "__main__":
    run()

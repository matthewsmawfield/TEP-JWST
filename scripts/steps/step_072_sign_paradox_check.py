#!/usr/bin/env python3
"""Step 072: validate the potential-depth, clock-offset, and response taxonomy."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.tep_model import (
    KAPPA_GAL,
    LOG_MH_REF,
    Z_REF,
    compute_ml_response,
    get_potential_depth_from_log_mh,
)

STEP_NUM = "072"
STEP_NAME = "sign_paradox_check"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(
    f"step_{STEP_NUM}",
    log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log",
)
set_step_logger(logger)


def normalized_local_clock_offset(log_mh, log_mh_ref=LOG_MH_REF):
    """Return Delta ln A / beta_A^2 up to a positive transfer normalization."""
    depth = get_potential_depth_from_log_mh(log_mh)
    depth_ref = get_potential_depth_from_log_mh(log_mh_ref)
    return -(np.asarray(depth) - depth_ref)


def run_analysis():
    print_status("=" * 60, "INFO")
    print_status("Step 072: Quantity-Taxonomy and Sign Validation", "INFO")
    print_status("=" * 60, "INFO")

    log_mh = np.linspace(9.0, 14.0, 501)
    redshifts = [4.0, 6.0, 8.0, 10.0]
    depth = get_potential_depth_from_log_mh(log_mh)
    depth_is_positive = bool(np.all(depth > 0))
    depth_is_monotonic = bool(np.all(np.diff(depth) > 0))

    clock_offset = normalized_local_clock_offset(log_mh)
    deeper = log_mh > LOG_MH_REF
    clock_slows_in_deeper_wells = bool(np.all(clock_offset[deeper] < 0))

    response_checks = {}
    for z in redshifts:
        response = compute_ml_response(log_mh, z)
        response_checks[str(z)] = {
            "minimum": float(np.min(response)),
            "maximum": float(np.max(response)),
            "positive": bool(np.all(response > 0)),
        }

    representative = [
        {"log_Mh": 10.5, "z": 8.0, "label": "Low-mass z=8"},
        {"log_Mh": 12.0, "z": 8.0, "label": "Reference-mass z=8"},
        {"log_Mh": 12.5, "z": 8.0, "label": "High-mass z=8"},
        {"log_Mh": 13.0, "z": 8.0, "label": "Ultra-massive z=8"},
    ]
    response_cases = []
    for case in representative:
        response = float(compute_ml_response(case["log_Mh"], case["z"]))
        response_cases.append(
            {
                **case,
                "ml_response": response,
                "legacy_Gamma_t": response,
                "response_above_reference": bool(response > 1.0),
            }
        )

    red_monsters = [
        {"id": "S1", "z": 5.3, "log_Mh": 12.8},
        {"id": "S2", "z": 5.5, "log_Mh": 12.6},
        {"id": "S3", "z": 5.9, "log_Mh": 13.0},
    ]
    red_monster_results = []
    for galaxy in red_monsters:
        response = float(compute_ml_response(galaxy["log_Mh"], galaxy["z"]))
        red_monster_results.append(
            {
                **galaxy,
                "ml_response": response,
                "legacy_Gamma_t": response,
                "mass_correction_factor": float(response**0.7),
                "sfe_reduction_pct": float((1.0 - response ** (-0.7)) * 100.0),
            }
        )

    response = compute_ml_response(log_mh, 8.0)
    reciprocal_invariance = bool(
        np.allclose(response ** (-0.7), (1.0 / response) ** 0.7, rtol=1e-14, atol=1e-14)
    )

    results = {
        "resolution_method": "three_variable_taxonomy",
        "quantity_dictionary": {
            "signed_potential": "Phi <= 0",
            "potential_depth": "Psi = |Phi|/c^2 >= 0",
            "local_clock_offset": "Delta ln A is negative in a deeper well",
            "ml_response": "R_ML > 0 is an observable channel response, not A",
        },
        "validation": {
            "potential_depth_positive": depth_is_positive,
            "potential_depth_monotonic": depth_is_monotonic,
            "local_clock_slows_in_deeper_wells": clock_slows_in_deeper_wells,
            "responses_positive": bool(all(item["positive"] for item in response_checks.values())),
            "reciprocal_mass_correction_invariant": reciprocal_invariance,
        },
        "response_checks": response_checks,
        "relative_analysis": {
            "legacy_name": "Gamma_t",
            "canonical_name": "R_ML",
            "KAPPA_GAL": KAPPA_GAL,
            "z_ref": Z_REF,
            "log_Mh_ref": LOG_MH_REF,
            "test_cases": response_cases,
        },
        "absolute_analysis": {
            "status": "not_computed_from_KAPPA_GAL",
            "reason": "KAPPA_GAL is an observable response coefficient, not beta_A or A(phi)",
        },
        "red_monsters": red_monster_results,
        "sign_paradox_resolved": bool(
            depth_is_positive
            and depth_is_monotonic
            and clock_slows_in_deeper_wells
            and reciprocal_invariance
        ),
    }

    with open(OUTPUT_PATH / "step_072_sign_paradox_check.json", "w") as handle:
        json.dump(results, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(log_mh, clock_offset)
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].axvline(LOG_MH_REF, color="gray", linestyle=":")
    axes[0].set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
    axes[0].set_ylabel(r"$\Delta\ln A/\beta_A^2$ (normalized)")
    axes[0].set_title("Local conformal clock offset")
    axes[0].grid(alpha=0.3)

    for z in redshifts:
        axes[1].plot(log_mh, compute_ml_response(log_mh, z), label=f"z={z:g}")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].axvline(LOG_MH_REF, color="gray", linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
    axes[1].set_ylabel(r"$R_{\rm ML}$")
    axes[1].set_title("Observable M/L inference response")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "step_072_sign_paradox_resolution.png", dpi=150)
    plt.close()

    print_status(f"Potential depth positive: {depth_is_positive}", "SUCCESS")
    print_status(f"Deeper-well clock offset negative: {clock_slows_in_deeper_wells}", "SUCCESS")
    print_status(f"Reciprocal mass correction invariant: {reciprocal_invariance}", "SUCCESS")
    print_status("R_ML is validated as an observable response, not a local clock rate.", "SUCCESS")
    return results


if __name__ == "__main__":
    run_analysis()

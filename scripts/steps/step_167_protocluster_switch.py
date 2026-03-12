import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import safe_json_default

STEP_NUM = "167"
STEP_NAME = "protocluster_switch"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"
for path in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

DJA_FILE = DATA_RAW / "dja_msaexp_emission_lines_v4.4.csv.gz"
COMBINED_SPEC_FILE = DATA_INTERIM / "combined_spectroscopic_catalog.csv"
BOOTSTRAP_N = 4000
RNG_SEED = 167


def _clip_p(value: float) -> float:
    return max(float(value), 1e-300)


def _stable_label_seed(label: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(label))


def _bootstrap_mean_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(BOOTSTRAP_N, dtype=float)
    for i in range(BOOTSTRAP_N):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        out[i] = float(np.mean(a[ia]) - np.mean(b[ib]))
    return out


def _assign_footprints(df: pd.DataFrame, gap_deg: float = 5.0) -> pd.DataFrame:
    out = df.sort_values("ra").reset_index(drop=True).copy()
    field_id = np.zeros(len(out), dtype=int)
    current = 0
    for i in range(1, len(out)):
        if out.loc[i, "ra"] - out.loc[i - 1, "ra"] > gap_deg:
            current += 1
        field_id[i] = current
    out["footprint_id"] = field_id
    return out


def _assign_named_footprints(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    labels = out[label_col].astype(str).fillna("unknown")
    unique_labels = sorted(labels.unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    out["footprint_id"] = labels.map(label_map).astype(int)
    out["footprint_label"] = labels
    return out


def _compute_density(df: pd.DataFrame, n_neighbor: int, z_window: float) -> np.ndarray:
    coords = df[["ra", "dec"]].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    footprint = df["footprint_id"].to_numpy(dtype=int)
    density = np.full(len(df), np.nan)
    for i in range(len(df)):
        mask = (footprint == footprint[i]) & (np.abs(z - z[i]) < z_window)
        mask[i] = False
        if mask.sum() < n_neighbor:
            continue
        nbr = coords[mask]
        dra = (nbr[:, 0] - coords[i, 0]) * np.cos(np.deg2rad(coords[i, 1]))
        ddec = nbr[:, 1] - coords[i, 1]
        dist = np.sqrt(dra * dra + ddec * ddec) * 60.0
        d_n = np.sort(dist)[n_neighbor - 1]
        if d_n > 0:
            density[i] = n_neighbor / (np.pi * d_n * d_n)
    return density


def _residualize(values: np.ndarray, controls: list[np.ndarray], footprint: np.ndarray | None = None) -> np.ndarray:
    matrices = [np.ones(len(values))]
    matrices.extend(np.asarray(c, dtype=float) for c in controls)
    if footprint is not None:
        dummies = pd.get_dummies(pd.Series(footprint).astype(int), drop_first=True)
        if dummies.shape[1] > 0:
            matrices.append(dummies.to_numpy(dtype=float))
    X = np.column_stack(matrices)
    beta = np.linalg.lstsq(X, np.asarray(values, dtype=float), rcond=None)[0]
    return np.asarray(values, dtype=float) - X @ beta


def _matched_cell_deltas(df: pd.DataFrame, value_col: str) -> list[float]:
    out = []
    if len(df) < 80:
        return out
    work = df.copy()
    work["mass_bin"] = pd.qcut(work["log_Mstar"], q=5, duplicates="drop")
    work["z_bin"] = pd.qcut(work["z"], q=4, duplicates="drop")
    grouped = work.groupby(["footprint_id", "mass_bin", "z_bin"], observed=False)
    for _, cell in grouped:
        if cell["environment"].nunique() < 2 or len(cell) < 12:
            continue
        dense = cell[cell["environment"] == "dense"][value_col]
        field = cell[cell["environment"] == "field"][value_col]
        if len(dense) < 4 or len(field) < 4:
            continue
        out.append(float(np.nanmedian(dense) - np.nanmedian(field)))
    return out


def _summarize_environment_contrast(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    *,
    within_footprint_quantiles: bool = False,
) -> dict | None:
    if len(df) < 40 or value_col not in df.columns:
        return None
    work = df.dropna(subset=[value_col, "density", "z", "log_Mstar", "footprint_id"]).copy()
    if len(work) < 40:
        return None
    if within_footprint_quantiles:
        density_q = work.groupby("footprint_id")["density"].quantile([0.25, 0.75]).unstack()
        work = work.join(density_q, on="footprint_id", rsuffix="_q")
        field = work[work["density"] <= work[0.25]].copy()
        dense = work[work["density"] >= work[0.75]].copy()
        low_thr = None
        high_thr = None
    else:
        low_thr = float(work["density"].quantile(0.25))
        high_thr = float(work["density"].quantile(0.75))
        field = work[work["density"] <= low_thr].copy()
        dense = work[work["density"] >= high_thr].copy()
    if len(field) < 20 or len(dense) < 20:
        return None
    field["environment"] = "field"
    dense["environment"] = "dense"
    contrast = pd.concat([field, dense], ignore_index=True)
    rng = np.random.default_rng(RNG_SEED + _stable_label_seed(label) % 1000)
    boot = _bootstrap_mean_delta(dense[value_col].to_numpy(dtype=float), field[value_col].to_numpy(dtype=float), rng)
    matched = _matched_cell_deltas(contrast, value_col)
    mw = mannwhitneyu(dense[value_col], field[value_col], alternative="two-sided")
    return {
        "label": label,
        "n_total": int(len(work)),
        "n_field": int(len(field)),
        "n_dense": int(len(dense)),
        "density_threshold_field_q25": low_thr,
        "density_threshold_dense_q75": high_thr,
        "density_split_method": (
            "within_footprint_q25_q75"
            if within_footprint_quantiles
            else "global_q25_q75"
        ),
        "field_median": float(np.nanmedian(field[value_col])),
        "dense_median": float(np.nanmedian(dense[value_col])),
        "field_mean": float(np.nanmean(field[value_col])),
        "dense_mean": float(np.nanmean(dense[value_col])),
        "delta_dense_minus_field": float(np.nanmean(dense[value_col]) - np.nanmean(field[value_col])),
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_one_sided_p_dense_less_than_field": _clip_p((float(np.sum(boot >= 0.0)) + 1.0) / (len(boot) + 1.0)),
        "mannwhitney_p_two_sided": _clip_p(mw.pvalue),
        "matched_cell_median_deltas_dense_minus_field": matched,
        "n_matched_cells": int(len(matched)),
        "n_matched_cells_supporting_tep": int(sum(delta < 0 for delta in matched)),
    }


def _load_dja_beta() -> tuple[pd.DataFrame | None, dict]:
    if not DJA_FILE.exists():
        return None, {"status": "missing_data", "file": str(DJA_FILE)}
    table = pd.read_csv(DJA_FILE, compression="infer", low_memory=False)
    z = pd.to_numeric(table.get("z_best"), errors="coerce")
    mass = pd.to_numeric(table.get("phot_mass"), errors="coerce")
    if np.nanmedian(mass.dropna()) > 100:
        mass = np.log10(mass.where(mass > 0))
    beta = pd.to_numeric(table.get("beta"), errors="coerce")
    ra = pd.to_numeric(table.get("ra"), errors="coerce")
    dec = pd.to_numeric(table.get("dec"), errors="coerce")
    grade = pd.to_numeric(table.get("grade"), errors="coerce") if "grade" in table.columns else pd.Series(np.nan, index=table.index)
    srcid = pd.to_numeric(table.get("srcid"), errors="coerce") if "srcid" in table.columns else pd.Series(np.arange(len(table)))
    root = table.get("root") if "root" in table.columns else pd.Series(np.nan, index=table.index)
    df = pd.DataFrame({
        "srcid": srcid,
        "ra": ra,
        "dec": dec,
        "z": z,
        "log_Mstar": mass,
        "beta": beta,
        "grade": grade,
        "root": root,
    })
    if "sn50" in table.columns:
        df["sn50"] = pd.to_numeric(table["sn50"], errors="coerce")
    df = df.dropna(subset=["ra", "dec", "z", "log_Mstar", "beta"]).copy()
    df = df[(df["z"] >= 5.0) & (df["z"] < 8.0) & df["log_Mstar"].between(5.0, 13.0)].copy()
    if df["grade"].notna().any():
        df = df[(df["grade"] >= 2) | df["grade"].isna()].copy()
    df = df.sort_values(["srcid", "grade", "sn50"] if "sn50" in df.columns else ["srcid", "grade"], ascending=[True, False, False] if "sn50" in df.columns else [True, False], na_position="last")
    df = df.drop_duplicates(subset="srcid", keep="first")
    df = df[df["beta"].between(-8.0, 3.0)].copy()
    root_nonnull_fraction = float(df["root"].notna().mean()) if "root" in df.columns else 0.0
    if "root" in df.columns and root_nonnull_fraction >= 0.8 and df["root"].nunique(dropna=True) >= 3:
        df = _assign_named_footprints(df.dropna(subset=["root"]).copy(), "root")
        footprint_method = "dja_root"
        min_footprint_size = 20
        n_neighbor = 3
    else:
        df = _assign_footprints(df)
        footprint_method = "ra_gap"
        min_footprint_size = 50
        n_neighbor = 5
    footprint_sizes = df["footprint_id"].value_counts()
    df = df[df["footprint_id"].isin(footprint_sizes[footprint_sizes >= min_footprint_size].index)].copy()
    df["density"] = _compute_density(df, n_neighbor=n_neighbor, z_window=0.15)
    df = df.dropna(subset=["density"]).copy()
    df["beta_resid_mass_z_field"] = _residualize(
        df["beta"].to_numpy(dtype=float),
        [df["z"].to_numpy(dtype=float), df["log_Mstar"].to_numpy(dtype=float)],
        footprint=df["footprint_id"].to_numpy(dtype=int),
    )
    return df, {
        "status": "success",
        "n_final": int(len(df)),
        "n_footprints": int(df["footprint_id"].nunique()),
        "footprint_sizes": {str(k): int(v) for k, v in df["footprint_id"].value_counts().sort_index().items()},
        "footprint_method": footprint_method,
        "density_n_neighbor": int(n_neighbor),
        "min_footprint_size": int(min_footprint_size),
        "observables_available": ["beta"],
        "observables_missing": ["balmer_absorption_eqw", "Dn4000"],
    }


def _load_jades_mwa_companion() -> tuple[pd.DataFrame | None, dict]:
    if not COMBINED_SPEC_FILE.exists():
        return None, {"status": "missing_data", "file": str(COMBINED_SPEC_FILE)}
    df = pd.read_csv(COMBINED_SPEC_FILE)
    df = df.rename(columns={"z_spec": "z"})
    df = df.dropna(subset=["ra", "dec", "z", "log_Mstar", "mwa"]).copy()
    df = df[(df["z"] >= 5.0) & (df["z"] < 8.0)].copy()
    if len(df) < 8:
        return None, {"status": "underpowered", "n": int(len(df))}
    df = _assign_footprints(df, gap_deg=20.0)
    df["density"] = _compute_density(df, n_neighbor=3, z_window=0.15)
    df = df.dropna(subset=["density"]).copy()
    if len(df) < 8:
        return None, {"status": "underpowered_after_density", "n": int(len(df))}
    df["mwa_resid_mass_z_field"] = _residualize(
        df["mwa"].to_numpy(dtype=float),
        [df["z"].to_numpy(dtype=float), df["log_Mstar"].to_numpy(dtype=float)],
        footprint=df["footprint_id"].to_numpy(dtype=int),
    )
    return df, {
        "status": "success",
        "n_final": int(len(df)),
        "observables_available": ["mwa"],
        "source_catalogs": sorted(df["source_catalog"].dropna().unique().tolist()) if "source_catalog" in df.columns else [],
    }


def _assessment(primary: dict | None, companion: dict | None) -> str:
    if primary is None:
        return "The protocluster-switch test could not be run with stable local environment and age-proxy inputs."
    delta = primary["delta_dense_minus_field"]
    ci = primary["bootstrap_ci_95"]
    if ci[1] < 0:
        return "The primary DJA beta test shows the sign-reversal predicted by TEP: overdense galaxies are bluer (younger) than matched field galaxies."
    if ci[0] > 0:
        return "The primary DJA beta test does not show the TEP sign reversal; the dense environments are slightly redder or statistically indistinguishable after mass+z+footprint control."
    if companion is not None and companion.get("delta_dense_minus_field", 0.0) < 0:
        return "The primary DJA beta test is mixed, but the low-N JADES spectroscopic-age companion points in the TEP sign-reversal direction."
    return "The current protocluster-switch implementation is mixed or null with existing local data: no clean dense-younger-than-field reversal emerges after matched controls."


def run():
    print_status(f"STEP {STEP_NUM}: Protocluster switch sign-reversal test", "INFO")
    dja_df, dja_meta = _load_dja_beta()
    jades_df, jades_meta = _load_jades_mwa_companion()
    dja_within_footprint = bool(dja_meta.get("footprint_method") == "dja_root") if isinstance(dja_meta, dict) else False
    primary = (
        _summarize_environment_contrast(
            dja_df,
            "beta_resid_mass_z_field",
            "dja_beta_residual",
            within_footprint_quantiles=dja_within_footprint,
        )
        if dja_df is not None else None
    )
    raw_beta = (
        _summarize_environment_contrast(
            dja_df,
            "beta",
            "dja_beta_raw",
            within_footprint_quantiles=dja_within_footprint,
        )
        if dja_df is not None else None
    )
    jades_age = _summarize_environment_contrast(jades_df, "mwa_resid_mass_z_field", "jades_mwa_residual") if jades_df is not None else None
    jades_age_raw = _summarize_environment_contrast(jades_df, "mwa", "jades_mwa_raw") if jades_df is not None else None
    result = {
        "step": STEP_NUM,
        "name": STEP_NAME,
        "status": "SUCCESS",
        "description": "Protocluster-switch sign-reversal test using existing DJA and JADES spectroscopic-era data products",
        "prediction_table": {
            "standard_physics": "At fixed stellar mass and redshift, overdense galaxies should appear older or at least not younger than field galaxies.",
            "tep": "At fixed stellar mass and redshift, screened overdense galaxies should appear younger than unscreened field galaxies.",
            "supportive_sign_for_beta_dense_minus_field": "negative",
            "supportive_sign_for_mwa_dense_minus_field": "negative",
        },
        "data_limits": {
            "dja": dja_meta,
            "jades_companion": jades_meta,
            "note": "Balmer absorption equivalent widths and Dn4000 are not present in the currently reproducible local DJA ingestion, so the live primary age proxy is UV slope beta.",
        },
        "primary_test": primary,
        "secondary_tests": {
            "dja_beta_raw": raw_beta,
            "jades_mwa_residual": jades_age,
            "jades_mwa_raw": jades_age_raw,
        },
    }
    result["assessment"] = _assessment(primary, jades_age)
    out_json = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    out_json.write_text(json.dumps(result, indent=2, default=safe_json_default))
    if primary is not None:
        print_status(
            f"Primary dense-field beta residual contrast Δ={primary['delta_dense_minus_field']:.3f} with 95% CI [{primary['bootstrap_ci_95'][0]:.3f}, {primary['bootstrap_ci_95'][1]:.3f}]",
            "INFO",
        )
    print_status(result["assessment"], "INFO")
    return result


if __name__ == "__main__":
    run()

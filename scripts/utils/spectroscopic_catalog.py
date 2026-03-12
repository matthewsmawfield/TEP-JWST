from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
RESULTS_INTERIM = PROJECT_ROOT / "results" / "interim"
DEFAULT_OUTPUT = DATA_INTERIM / "combined_spectroscopic_catalog.csv"


def _to_native(arr) -> np.ndarray:
    out = np.array(arr)
    if out.dtype.byteorder == ">":
        out = out.byteswap().view(out.dtype.newbyteorder("="))
    return out


def _load_jades_coord_table() -> pd.DataFrame:
    candidates = [
        DATA_RAW / "JADES_z_gt_8_Candidates_Hainline_et_al.fits",
        DATA_RAW / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with fits.open(path, memmap=False) as hdul:
            ext_name = "CIRC_CONV" if "CIRC_CONV" in hdul else ("CIRC" if "CIRC" in hdul else None)
            if ext_name is None:
                continue
            data = hdul[ext_name].data
            cols = getattr(data, "columns", None)
            names = cols.names if cols is not None else []
            if not all(name in names for name in ["ID", "RA", "DEC"]):
                continue
            df = pd.DataFrame(
                {
                    "ID": pd.to_numeric(_to_native(data["ID"]), errors="coerce"),
                    "ra": _to_native(data["RA"]).astype(float),
                    "dec": _to_native(data["DEC"]).astype(float),
                }
            )
            return df.dropna(subset=["ID", "ra", "dec"]).drop_duplicates(subset=["ID"])
    return pd.DataFrame(columns=["ID", "ra", "dec"])


def _load_jades_physical_with_coords() -> pd.DataFrame:
    phys_path = DATA_INTERIM / "jades_highz_physical.csv"
    if not phys_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(phys_path)
    if "ID" not in df.columns:
        return pd.DataFrame()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    coords = _load_jades_coord_table()
    if coords.empty:
        return pd.DataFrame()
    merged = df.merge(coords, on="ID", how="left")
    return merged.dropna(subset=["ra", "dec", "t_stellar_Gyr", "log_Mstar"])


def _load_uncover_spec() -> pd.DataFrame:
    path = DATA_RAW / "uncover" / "UNCOVER_DR4_SPS_zspec_catalog.fits"
    if not path.exists():
        return pd.DataFrame()
    tab = Table.read(path)
    df = tab.to_pandas()
    if not {"z_spec", "flag_zspec_qual", "mstar_50", "mwa_50"}.issubset(df.columns):
        return pd.DataFrame()
    mask = (df["z_spec"] >= 4) & (df["flag_zspec_qual"] >= 2)
    df = df.loc[mask].copy()
    rename_map = {
        "id_msa": "id",
        "ra": "ra",
        "dec": "dec",
        "z_spec": "z_spec",
        "mstar_50": "log_Mstar",
        "mwa_50": "mwa",
        "dust2_50": "dust",
        "met_50": "met",
        "sfr100_50": "sfr",
    }
    df = df.rename(columns=rename_map)
    for col in ["id", "ra", "dec", "z_spec", "log_Mstar", "mwa", "dust", "met", "sfr"]:
        if col not in df.columns:
            df[col] = np.nan
    df["source_catalog"] = "UNCOVER"
    return df[["id", "ra", "dec", "z_spec", "log_Mstar", "mwa", "dust", "met", "sfr", "source_catalog"]].dropna(subset=["z_spec", "log_Mstar", "mwa"])


def _load_jades_spec_fallback() -> pd.DataFrame:
    spec_path = RESULTS_INTERIM / "step_149_jades_dr4_specz.csv"
    if not spec_path.exists():
        return pd.DataFrame()
    spec = pd.read_csv(spec_path)
    needed = {"unique_id", "ra", "dec", "z_spec", "log_Mstar"}
    if not needed.issubset(spec.columns):
        return pd.DataFrame()
    spec = spec.dropna(subset=["ra", "dec", "z_spec", "log_Mstar"]).copy()
    spec = spec[spec["z_spec"] >= 4]
    if spec.empty:
        return pd.DataFrame()

    phys = _load_jades_physical_with_coords()
    if phys.empty:
        return pd.DataFrame()

    sc_spec = SkyCoord(spec["ra"].values * u.deg, spec["dec"].values * u.deg)
    sc_phys = SkyCoord(phys["ra"].values * u.deg, phys["dec"].values * u.deg)
    idx, d2d, _ = sc_spec.match_to_catalog_sky(sc_phys)
    mask = d2d < 0.1 * u.arcsec
    if not np.any(mask):
        return pd.DataFrame()

    matched_spec = spec.loc[mask].copy().reset_index(drop=True)
    matched_phys = phys.iloc[idx[mask]].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "id": matched_spec["unique_id"],
            "ra": matched_spec["ra"],
            "dec": matched_spec["dec"],
            "z_spec": matched_spec["z_spec"],
            "log_Mstar": matched_phys["log_Mstar"].fillna(matched_spec["log_Mstar"]),
            "mwa": matched_phys["t_stellar_Gyr"],
            "dust": np.nan,
            "met": np.nan,
            "sfr": np.nan,
            "source_catalog": "JADES_DR4_x_PHYS",
        }
    )
    return out.dropna(subset=["z_spec", "log_Mstar", "mwa"])


def build_combined_spectroscopic_catalog() -> pd.DataFrame:
    frames = [frame for frame in [_load_uncover_spec(), _load_jades_spec_fallback()] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["id", "ra", "dec", "z_spec", "log_Mstar", "mwa", "dust", "met", "sfr", "source_catalog"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["id", "source_catalog"])


def ensure_combined_spectroscopic_catalog(output_path: Path | None = None) -> Path | None:
    output = output_path or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return output
    combined = build_combined_spectroscopic_catalog()
    if combined.empty:
        return None
    combined.to_csv(output, index=False)
    return output

#!/usr/bin/env python3
"""
TEP-JWST Step 163: External dataset shortlist and ingestion registry for future replication

External dataset shortlist and ingestion registry for future replication


Author: Matthew L. Smawfield
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM  = "163"
STEP_NAME = "external_dataset_registry"

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH   = PROJECT_ROOT / "logs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

from pathlib import Path as _Path

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INT = PROJECT_ROOT / "data" / "interim"

DATASETS = [
    {
        "name":    "UNCOVER DR4 Full SPS Catalog",
        "file":    "uncover/UNCOVER_DR4_SPS_catalog.fits",
        "url":     "https://jwst-uncover.github.io/DR4.html",
        "ref":     "Weibel et al. (2024), Wang et al. (2024)",
        "n_sources": 74020,
        "used_in": ["step_001", "step_020", "step_152", "step_162"],
        "pipeline_step": "step_001",
    },
    {
        "name":    "UNCOVER DR4 Spec-z SPS Catalog",
        "file":    "uncover/UNCOVER_DR4_SPS_zspec_catalog.fits",
        "url":     "https://drive.google.com/file/d/1j32n3e7hX0iw5ZyGlVAbyIf4MmjM4RfS/view?usp=drive_link",
        "ref":     "Price et al. (2025), Wang et al. (2024)",
        "n_sources": 668,
        "used_in": ["step_152"],
        "pipeline_step": "step_152",
    },
    {
        "name":    "CEERS NIRCam Photometric Catalog v1.0",
        "file":    "ceers_cat_v1.0.fits.gz",
        "url":     "https://ceers.github.io/releases.html",
        "ref":     "Finkelstein et al. (2023), Bagley et al. (2023)",
        "n_sources": None,
        "used_in": ["step_031", "step_032"],
        "pipeline_step": "step_031",
    },
    {
        "name":    "COSMOS-Web LePHARE Catalog v1 (COSMOS2025)",
        "file":    "COSMOSWeb_mastercatalog_v1_lephare.fits",
        "url":     "https://cosmos2025.iap.fr/",
        "ref":     "Shuntov et al. (2025)",
        "n_sources": 784016,
        "used_in": ["step_033", "step_034", "step_153", "step_157"],
        "pipeline_step": "step_033",
        "requires_auth": True,
    },
    {
        "name":    "JADES GOODS-S Deep Photometry v2.0",
        "file":    "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
        "url":     "https://archive.stsci.edu/hlsp/jades",
        "ref":     "Rieke et al. (2023), Eisenstein et al. (2023)",
        "n_sources": None,
        "used_in": ["step_149", "step_156"],
        "pipeline_step": "step_149",
    },
    {
        "name":    "JADES DR4 Spectroscopic Catalog",
        "file":    "jades_hainline/JADES_DR4_spectroscopic_catalog.fits",
        "url":     "https://jades-survey.github.io/",
        "ref":     "D'Eugenio et al. (2025), Hainline et al. (2024)",
        "n_sources": 5190,
        "used_in": ["step_149", "step_154", "step_158"],
        "pipeline_step": "step_149",
    },
    {
        "name":    "JADES DR5 GOODS-S Photometric Catalog v5.0",
        "file":    "jades_hainline/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits",
        "file_candidates": [
            "jades_hainline/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits",
            "JADES_z_gt_8_Candidates_Hainline_et_al.fits",
        ],
        "url":     "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits",
        "ref":     "Robertson et al. (2026)",
        "n_sources": 94000,
        "used_in": ["step_155"],
        "pipeline_step": "step_155",
    },
    {
        "name":    "Kokorev LRD Catalog v1.1",
        "file":    "kokorev_lrd_catalog_v1.1.fits",
        "url":     "https://arxiv.org/abs/2401.09981",
        "ref":     "Kokorev et al. (2024)",
        "n_sources": None,
        "used_in": ["step_042"],
        "pipeline_step": "step_042",
    },
    {
        "name":    "Labbe et al. (2023) Massive Galaxy Sample",
        "file":    "labbe_2023_github.tar.gz",
        "url":     "https://github.com/ivelascog/labbe2023",
        "ref":     "Labbe et al. (2023), Nature",
        "n_sources": 6,
        "used_in": ["step_050"],
        "pipeline_step": "step_050",
    },
    {
        "name":    "DJA NIRSpec Merged v4.4",
        "file":    "dja_msaexp_emission_lines_v4.4.csv.gz",
        "file_candidates": [
            "dja_msaexp_emission_lines_v4.4.csv.gz",
            "dja_nirspec_merged_v4.4.fits",
        ],
        "url":     "https://zenodo.org/records/15472354/files/dja_msaexp_emission_lines_v4.4.csv.gz?download=1",
        "ref":     "Brammer et al. (2024), de Graaff et al. (2024)",
        "n_sources": 80367,
        "used_in": ["step_150", "step_158"],
        "pipeline_step": "step_150",
        "note": "Public merged table hosted on Zenodo; spectra bundles remain available via the DJA archive",
    },
]


def check_availability():
    """Check which datasets are locally available."""
    for ds in DATASETS:
        candidates = ds.get("file_candidates", [ds["file"]])
        matched_path = None
        for rel_path in candidates:
            fpath = DATA_RAW / rel_path
            if fpath.exists():
                matched_path = fpath
                break
        ds["local_available"] = matched_path is not None
        if matched_path is not None:
            ds["local_size_mb"] = round(matched_path.stat().st_size / 1e6, 1)
            ds["local_file_detected"] = str(matched_path.relative_to(DATA_RAW))
        else:
            ds["local_size_mb"] = None
            ds["local_file_detected"] = None
    return DATASETS


def run():
    print_status(f"STEP {STEP_NUM}: External dataset registry", "INFO")

    datasets = check_availability()
    n_available   = sum(1 for d in datasets if d.get("local_available"))
    n_missing     = sum(1 for d in datasets if not d.get("local_available"))
    n_total       = len(datasets)

    for ds in datasets:
        status = "PRESENT" if ds.get("local_available") else "MISSING"
        logger.info(f"  [{status}] {ds['name']}")

    result = {
        "step":   STEP_NUM,
        "name":   STEP_NAME,
        "status": "SUCCESS",
        "description": "External dataset registry for TEP-JWST pipeline",
        "n_datasets_total":     n_total,
        "n_datasets_available": n_available,
        "n_datasets_missing":   n_missing,
        "datasets": datasets,
    }

    out_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {out_path}")
    print_status(
        f"Step {STEP_NUM} complete. {n_available}/{n_total} datasets available locally.", "INFO"
    )
    return result


main = run

if __name__ == "__main__":
    run()

import numpy as np
from astropy.io import fits
import sys

def count_valid_spec(path, col_name, ext=1):
    try:
        with fits.open(path) as hdul:
            data = hdul[ext].data
            if col_name in data.columns.names:
                valid = data[col_name] > 0
                count = np.sum(valid)
                print(f"{path} [{col_name}]: {count} valid")
                return count
            else:
                print(f"{path}: Column {col_name} not found")
                return 0
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0

print("--- Counting Spectroscopic Redshifts ---")
# UNCOVER
count_valid_spec("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits", "z_spec")
count_valid_spec("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits", "z_spec")

# JADES Hainline
# In Hainline, z_spec might be -1 or NaN if missing.
try:
    path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    with fits.open(path) as hdul:
        data = hdul['PRIMARY_SAMPLE'].data
        if 'z_spec' in data.columns.names:
            # Check for valid z_spec (often > 0 or not NaN)
            z = data['z_spec']
            valid = (z > 0) & np.isfinite(z)
            print(f"JADES Hainline [z_spec]: {np.sum(valid)} valid")
            # Also check z_spec_source
            sources = data['z_spec_source'][valid]
            print(f"Sources: {np.unique(sources)}")
except Exception as e:
    print(f"Error JADES Hainline: {e}")

print("\n--- Checking for Resolved Data (Apertures) ---")
# JADES CIRC
try:
    path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            cols = hdul['CIRC'].data.columns.names
            print(f"JADES GOODS-S CIRC columns: {cols[:20]}")
except Exception as e:
    print(f"Error JADES CIRC: {e}")

# CEERS
try:
    path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/ceers_cat_v1.0.fits"
    with fits.open(path) as hdul:
        # Check for aperture flux columns?
        cols = hdul[1].data.columns.names
        flux_cols = [c for c in cols if 'FLUX' in c]
        print(f"CEERS Flux columns (sample): {flux_cols[:20]}")
except Exception as e:
    print(f"Error CEERS: {e}")

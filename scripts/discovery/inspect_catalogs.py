import os
from astropy.io import fits
from astropy.table import Table
import numpy as np

def inspect_fits(filepath, name):
    print(f"\n--- Inspecting {name} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        if filepath.endswith('.gz'):
            # Gzip handling might be automatic in astropy, but let's be safe
            pass
            
        with fits.open(filepath) as hdul:
            print(f"HDU List: {[hdu.name for hdu in hdul]}")
            
            # Usually the catalog is in the 1st extension
            data = None
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    data = hdu.data
                    header = hdu.header
                    print(f"Table found in {hdu.name}")
                    break
            
            if data is None:
                print("No binary table found.")
                return

            columns = data.columns.names
            print(f"Total rows: {len(data)}")
            print(f"Total columns: {len(columns)}")
            
            # 1. Spectroscopic Data Search
            z_cols = [c for c in columns if 'z' in c.lower() or 'spec' in c.lower()]
            print(f"Potential redshift columns: {z_cols[:10]} ...")
            
            z_spec_col = None
            if 'z_spec' in columns:
                z_spec_col = 'z_spec'
            elif 'zspec' in columns:
                z_spec_col = 'zspec'
            elif 'ZSPEC' in columns:
                z_spec_col = 'ZSPEC'
            
            if z_spec_col:
                valid_z = data[z_spec_col] > 0
                n_spec = np.sum(valid_z)
                print(f"Found '{z_spec_col}' column. Valid entries (>0): {n_spec}")
            else:
                print("No obvious z_spec column found.")

            # 2. Morphology/Resolved Search
            morph_cols = [c for c in columns if any(x in c.lower() for x in ['rad', 'radius', 'sersic', 'n', 're', 'grad', 'flux_radius'])]
            print(f"Potential morphology columns: {morph_cols[:10]} ...")

            # 3. Environment Search
            env_cols = [c for c in columns if any(x in c.lower() for x in ['dens', 'env', 'cluster', 'group', 'neigh'])]
            print(f"Potential environment columns: {env_cols[:10]} ...")

    except Exception as e:
        print(f"Error inspecting {name}: {e}")

# Define paths
paths = {
    "UNCOVER_SPEC": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits",
    "UNCOVER_PHOT": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits",
    "CEERS": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/ceers_cat_v1.0.fits",
    "COSMOS": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/COSMOSWeb_mastercatalog_v1_lephare.fits",
    "JADES_GOODS_S": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    "JADES_GOODS_N": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits"
}

for name, path in paths.items():
    if "JADES" in name:
        # For JADES, we know there are specific extensions
        print(f"\n--- Inspecting {name} Detailed ---")
        try:
            with fits.open(path) as hdul:
                print(f"HDU List: {[hdu.name for hdu in hdul]}")
                # Inspect SIZE and PHOTOZ/EAZY
                for ext in ['SIZE', 'CIRC', 'KRON', 'PHOTOZ']:
                    if ext in hdul:
                        print(f"  Extension {ext}:")
                        data = hdul[ext].data
                        cols = data.columns.names
                        print(f"    Rows: {len(data)}")
                        print(f"    Cols: {cols[:10]} ...")
                        # Check for z_spec in any likely extension
                        z_cols = [c for c in cols if 'z' in c.lower() or 'spec' in c.lower()]
                        if z_cols:
                            print(f"    Potential z columns: {z_cols}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        inspect_fits(path, name)

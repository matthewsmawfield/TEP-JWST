import sys
from astropy.io import fits

files = [
    "/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits",
    "/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits"
]

print("Starting inspection...")
sys.stdout.flush()

for f in files:
    print(f"--- {f.split('/')[-1]} ---")
    try:
        with fits.open(f) as hdul:
            # Try extension 1 first
            ext = 1
            if len(hdul) < 2:
                ext = 0
            
            data = hdul[ext].data
            cols = data.columns.names
            print(f"Columns in ext {ext}: {cols}")
            
            # Check specific columns
            for check in ['id', 'ID', 'Id', 'z_spec', 'mstar_50', 'mwa_50', 'dust2_50', 'met_50']:
                if check in cols:
                    print(f"  Found: {check}")
                else:
                    # Case insensitive check
                    lower_cols = [c.lower() for c in cols]
                    if check.lower() in lower_cols:
                        actual = cols[lower_cols.index(check.lower())]
                        print(f"  Found (case mismatch): {check} -> {actual}")
                    else:
                        print(f"  MISSING: {check}")
    except Exception as e:
        print(f"Error: {e}")
    sys.stdout.flush()

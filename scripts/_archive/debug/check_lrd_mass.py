
from astropy.io import fits

def check_cols(path):
    print(f"Checking {path}")
    with fits.open(path) as hdul:
        cols = hdul[1].columns.names
        mass_cols = [c for c in cols if 'mass' in c.lower() or 'mstar' in c.lower() or 'logm' in c.lower()]
        print("Mass Columns found:", mass_cols)

check_cols("/Users/matthewsmawfield/www/TEP-JWST/data/raw/kokorev_lrd_catalog_v1.1.fits")

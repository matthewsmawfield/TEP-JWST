from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

try:
    with fits.open(path) as hdul:
        if 'SIZE' in hdul:
            print("--- SIZE Extension Columns ---")
            print(hdul['SIZE'].columns.names)
except Exception as e:
    print(f"Error: {e}")

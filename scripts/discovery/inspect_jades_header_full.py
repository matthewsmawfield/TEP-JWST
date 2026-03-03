from astropy.io import fits

files = [
    "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"
]

for f in files:
    print(f"--- {f.split('/')[-1]} ---")
    try:
        with fits.open(f) as hdul:
            if 'CIRC' in hdul:
                print("Header keys in CIRC extension:")
                header = hdul['CIRC'].header
                for k in header.keys():
                    print(f"{k}: {header[k]}")
    except Exception as e:
        print(f"Error: {e}")

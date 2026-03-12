from astropy.io import fits

files = [
    "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits"
]

for f in files:
    print(f"--- {f.split('/')[-1]} ---")
    try:
        with fits.open(f) as hdul:
            if 'CIRC' in hdul:
                header = hdul['CIRC'].header
                # Look for aperture size keywords
                for k in header.keys():
                    if 'APER' in k or 'RAD' in k or 'CIRC' in k:
                        print(f"{k}: {header[k]}")
    except Exception as e:
        print(f"Error: {e}")

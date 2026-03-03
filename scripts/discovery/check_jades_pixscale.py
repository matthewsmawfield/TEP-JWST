from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

with fits.open(path) as hdul:
    if 'CIRC' in hdul:
        header = hdul['CIRC'].header
        # Look for pixel scale
        print("Pixel Scale keywords:")
        for k in header.keys():
            if 'PIX' in k or 'SCALE' in k:
                print(f"{k} = {header[k]}")
        
        # Check comments for aperture columns
        print("\nColumn Comments:")
        for i in range(7): # Check first few CIRC columns
            key = f"TTYPE{i+1}"
            if key in header:
                print(f"{header[key]}: {header.comments[key]}")

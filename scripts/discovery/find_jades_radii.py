from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

try:
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            print("--- Checking for aperture sizes ---")
            # Usually defined in comments or specific keys like APER_0, RADIUS_0 etc.
            # Let's dump the full header to a string and search
            hdr_str = header.tostring(sep='\n')
            
            keywords = ['PIX', 'SCALE', 'RADIUS', 'APER', 'DIAM', 'CIRC']
            for line in hdr_str.split('\n'):
                for k in keywords:
                    if k in line:
                        print(line)
except Exception as e:
    print(f"Error: {e}")

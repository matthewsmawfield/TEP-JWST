from astropy.io import fits
import sys

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

try:
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            # Write header to a file
            with open("jades_circ_header.txt", "w") as f:
                f.write(header.tostring(sep='\n'))
            print("Header written to jades_circ_header.txt")
except Exception as e:
    print(f"Error: {e}")

import sys
from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

print(f"Inspecting {path} for aperture definitions...")
sys.stdout.flush()

try:
    with fits.open(path) as hdul:
        # Check Primary header for general info
        print("\n--- Primary Header ---")
        print(repr(hdul[0].header))
        
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            print("\n--- CIRC Header Keys ---")
            for k, v in header.items():
                if 'RAD' in k or 'APER' in k or 'CIRC' in str(v) or 'PIX' in k:
                    print(f"{k} = {v}")
            
            # Sometimes aperture sizes are in a separate extension or documented in comments
            # Let's check if there is an extension describing apertures
            print(f"\nExtensions: {[h.name for h in hdul]}")
            
except Exception as e:
    print(f"Error: {e}")
sys.stdout.flush()

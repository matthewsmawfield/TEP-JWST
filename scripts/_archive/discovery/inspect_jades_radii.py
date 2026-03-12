from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

try:
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            # Print first 100 cards to see if aperture sizes are defined in comments or keys
            print("--- Header Cards (First 50) ---")
            for i, card in enumerate(header.cards):
                if i > 50: break
                print(card)
            
            # Search for specific radius related keywords
            print("\n--- Radius Keywords ---")
            for k in header.keys():
                if 'RAD' in k or 'APER' in k or 'CIRC' in str(header[k]):
                    print(f"{k} = {header[k]}")
except Exception as e:
    print(f"Error: {e}")

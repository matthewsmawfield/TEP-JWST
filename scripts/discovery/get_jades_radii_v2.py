from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

with fits.open(path) as hdul:
    if 'CIRC' in hdul:
        header = hdul['CIRC'].header
        print("--- JADES CIRC Header Aperture Definitions ---")
        # Look for keywords that might define the aperture sizes
        # Common keys: APER_0, APER_1, RADIUS_0, etc.
        # Or in the comments of the TTYPE keys.
        
        for i in range(7): # CIRC0 to CIRC6
            key = f"TTYPE{368 + i}" # F606W_CIRC0 is around 368 based on previous output
            # Actually let's just search by value
            for card in header.cards:
                if f"CIRC{i}" in str(card.value):
                    print(card)
                    
        # Also check COMMENT and HISTORY
        print("\n--- Comments/History ---")
        if 'COMMENT' in header:
            print(header['COMMENT'])
        
        # Check standard aperture keys
        for k in header.keys():
            if k.startswith('APER'):
                print(f"{k} = {header[k]}")
                
        # Check pixel scale
        if 'PIXSCALE' in header:
            print(f"PIXSCALE = {header['PIXSCALE']}")

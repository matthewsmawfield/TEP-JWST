from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

try:
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            # Check comments for TTYPEs related to CIRC0
            # We saw TTYPE165 = F090W_CIRC0_ei in previous output
            
            print("Checking comments for CIRC columns...")
            for i in range(1, 200): # Scan first 200 keywords
                key = f"TTYPE{i}"
                if key in header:
                    val = header[key]
                    if 'CIRC' in val:
                        comment = header.comments[key]
                        print(f"{key} = {val} / {comment}")
                        
            # Also check if there's a specific APER keywords in primary or other headers
            for h in hdul:
                print(f"\n--- Extension {h.name} ---")
                for k in h.header:
                    if 'APER' in k or 'RAD' in k:
                        print(f"{k} = {h.header[k]}")

except Exception as e:
    print(f"Error: {e}")

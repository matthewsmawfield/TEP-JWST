from astropy.io import fits
import sys

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"

print(f"Checking {path} for aperture definitions...")
sys.stdout.flush()

try:
    with fits.open(path) as hdul:
        if 'CIRC' in hdul:
            header = hdul['CIRC'].header
            
            # 1. Look for 'APER' keywords
            print("\n--- APER Keywords ---")
            found = False
            for k, v in header.items():
                if 'APER' in k:
                    print(f"{k} = {v} / {header.comments[k]}")
                    found = True
            if not found:
                print("No APER keywords found.")

            # 2. Look for 'RAD' keywords
            print("\n--- RAD Keywords ---")
            found = False
            for k, v in header.items():
                if 'RAD' in k:
                    print(f"{k} = {v} / {header.comments[k]}")
                    found = True
            if not found:
                print("No RAD keywords found.")
                
            # 3. Look at HISTORY or COMMENT
            print("\n--- Comments/History ---")
            if 'HISTORY' in header:
                print("HISTORY found (showing last 10):")
                for line in list(header['HISTORY'])[-10:]:
                    print(line)
            
            # 4. Check for PIXSCALE
            if 'PIXSCALE' in header:
                print(f"\nPIXSCALE = {header['PIXSCALE']}")
            
            # 5. List first few column names to see if they imply sizes
            data = hdul['CIRC'].data
            print(f"\nFirst 10 columns: {data.columns.names[:10]}")

except Exception as e:
    print(f"Error: {e}")
sys.stdout.flush()

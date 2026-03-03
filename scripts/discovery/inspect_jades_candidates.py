import sys
from astropy.io import fits

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits"
print(f"--- Inspecting {path} ---")
sys.stdout.flush()

try:
    with fits.open(path) as hdul:
        print(f"HDU List: {[hdu.name for hdu in hdul]}")
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
                data = hdu.data
                cols = data.columns.names
                print(f"Extension: {hdu.name}")
                print(f"Rows: {len(data)}")
                print(f"Cols: {cols}")
except Exception as e:
    print(f"Error: {e}")
sys.stdout.flush()

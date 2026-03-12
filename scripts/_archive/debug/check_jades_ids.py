from astropy.io import fits
import pandas as pd

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits"
with fits.open(path) as hdul:
    data = hdul['PRIMARY_SAMPLE'].data
    ids = data['JADES_ID']
    print(f"Hainline IDs (first 5): {ids[:5]}")
    
    # Check z_spec distribution
    z_spec = data['z_spec']
    valid = z_spec > 0
    print(f"Hainline valid z_spec count: {sum(valid)}")
    print(f"Hainline z_spec (first 5 valid): {z_spec[valid][:5]}")

# Check physical catalog IDs
phys_path = "/Users/matthewsmawfield/www/TEP-JWST/data/interim/jades_highz_physical.csv"
df_phys = pd.read_csv(phys_path)
print(f"Physical IDs (first 5): {df_phys['ID'].head().tolist()}")

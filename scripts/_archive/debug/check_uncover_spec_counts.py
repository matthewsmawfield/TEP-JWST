from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np

path = "/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits"
# Use Table.read to handle FITS to Pandas conversion safely (handles endianness)
tab = Table.read(path)
df = tab.to_pandas()

print(f"Total rows: {len(df)}")
print(f"Valid z_spec > 0: {np.sum(df['z_spec'] > 0)}")
print(f"z_spec > 4: {np.sum(df['z_spec'] > 4)}")

# Check flags
# flag_zspec_qual: 1=low, 2=likely, 3=secure
print("\nQuality Flag Distribution (for z > 4):")
highz = df[df['z_spec'] > 4]
print(highz['flag_zspec_qual'].value_counts().sort_index())

print("\nQuality Flag Distribution (all):")
print(df['flag_zspec_qual'].value_counts().sort_index())

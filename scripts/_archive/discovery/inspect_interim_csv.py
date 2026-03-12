import pandas as pd
path = "/Users/matthewsmawfield/www/TEP-JWST/data/interim/uncover_highz_sed_properties.csv"
df = pd.read_csv(path)
print(f"Columns: {list(df.columns)}")

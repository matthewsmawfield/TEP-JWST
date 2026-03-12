import pandas as pd
import numpy as np
from pathlib import Path
from astropy.table import Table
import warnings
from astropy.units import UnitsWarning

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

def load_uncover_spec():
    """Load UNCOVER spectroscopic data."""
    path = DATA_RAW / "uncover" / "UNCOVER_DR4_SPS_zspec_catalog.fits"
    if not path.exists():
        print(f"UNCOVER spec file not found: {path}")
        return pd.DataFrame()
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)
        tab = Table.read(path)
    
    df = tab.to_pandas()
    # Filter high quality and high z
    # flag_zspec_qual: 1=low, 2=likely, 3=secure
    mask = (df['z_spec'] >= 4) & (df['flag_zspec_qual'] >= 2)
    df = df[mask].copy()
    
    # Standardize columns
    # We need: id, ra, dec, z_spec, log_Mstar, mwa (mass-weighted age), dust, met, source
    
    # Rename mapping
    # id is not in zspec catalog, but id_msa is. Or id_DR3.
    # Let's use id_msa as the primary ID for this catalog.
    rename_map = {
        'id_msa': 'id',
        'ra': 'ra',
        'dec': 'dec',
        'z_spec': 'z_spec',
        'mstar_50': 'log_Mstar',
        'mwa_50': 'mwa',       # Gyr
        'dust2_50': 'dust',    # Av? dust2 is usually Av or similar in Prospector
        'met_50': 'met',       # log(Z/Zsun)
        'sfr100_50': 'sfr'
    }
    
    # Check if columns exist
    for col in rename_map.keys():
        if col not in df.columns:
            print(f"Warning: UNCOVER column {col} not found")
            
    df = df.rename(columns=rename_map)
    df['source_catalog'] = 'UNCOVER'
    
    # Select only needed columns
    cols = ['id', 'ra', 'dec', 'z_spec', 'log_Mstar', 'mwa', 'dust', 'met', 'source_catalog']
    df = df[cols]
    
    # Filter valid physical params
    df = df.dropna(subset=['log_Mstar', 'mwa'])
    
    print(f"Loaded {len(df)} UNCOVER spectroscopic sources (z>4, Q>=2)")
    return df

def load_jades_hainline_spec():
    """Load JADES Hainline candidates with spectroscopic confirmation."""
    path = DATA_RAW / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    if not path.exists():
        print(f"JADES Hainline file not found: {path}")
        return pd.DataFrame()
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)
        tab = Table.read(path)
        
    df = tab.to_pandas()
    
    # Filter for valid z_spec
    # z_spec > 0
    mask = (df['z_spec'] > 0)
    df_spec = df[mask].copy()
    
    # Decode IDs if they are bytes
    if len(df_spec) > 0 and isinstance(df_spec['JADES_ID'].iloc[0], bytes):
        df_spec['JADES_ID'] = df_spec['JADES_ID'].apply(lambda x: x.decode('utf-8'))
    
    print(f"JADES Hainline: {len(df_spec)} sources with valid z_spec > 0")
    if len(df_spec) > 0:
        print(f"Sample IDs: {df_spec['JADES_ID'].head().tolist()}")

    # Hainline catalog mainly provides photometry and z_spec, need physical properties?
    # The physical properties might be in the derived catalog or need to be matched.
    # Interim file jades_highz_physical.csv has physical props but mostly photometric z.
    
    # Let's see if we can match with jades_highz_physical.csv based on ID or coords
    physical_path = DATA_INTERIM / "jades_highz_physical.csv"
    if physical_path.exists():
        df_phys = pd.read_csv(physical_path)
        print(f"Physical catalog: {len(df_phys)} sources")
        print(f"Physical IDs sample: {df_phys['ID'].head().tolist()}")
        
        # Match on JADES_ID
        # Hainline ID: JADES_ID (string)
        # Phys ID: ID (string)
        
        # Check intersection
        common_ids = set(df_spec['JADES_ID']).intersection(set(df_phys['ID']))
        print(f"Common IDs: {len(common_ids)}")
        
        # Merge
        merged = pd.merge(df_spec, df_phys, left_on='JADES_ID', right_on='ID', how='inner', suffixes=('', '_phys'))
        
        # If z_spec in phys is empty, use Hainline z_spec
        # Actually df_phys might have z_spec from other sources, but let's prioritize the Hainline catalog z_spec if it's the source
        
        # Prepare output
        out_df = pd.DataFrame()
        out_df['id'] = merged['JADES_ID']
        out_df['ra'] = merged['RA']
        out_df['dec'] = merged['DEC']
        out_df['z_spec'] = merged['z_spec']
        out_df['log_Mstar'] = merged['log_Mstar']
        out_df['mwa'] = merged['t_stellar_Gyr'] # Assuming t_stellar_Gyr is mass-weighted age or similar
        # dust/met might not be in physical csv?
        # Check columns of jades_highz_physical.csv:
        # ID,RA,DEC,z_phot,z_phot_lo,z_phot_hi,z_spec,MUV,mag_F277W,P_z_gt_7,delta_chisq,z_best,log_Mstar,log_Mstar_err,log_Mhalo,t_stellar_Gyr,t_stellar_err_Gyr,t_cosmic_Gyr,age_excess_Gyr
        # It lacks dust and met.
        
        out_df['dust'] = np.nan # Placeholder
        out_df['met'] = np.nan  # Placeholder
        out_df['source_catalog'] = 'JADES_Hainline'
        
        print(f"Loaded {len(out_df)} JADES Hainline spectroscopic sources (matched with physical params)")
        return out_df
    else:
        print("JADES physical parameters file missing, cannot use JADES spec for physical validation")
        return pd.DataFrame()

def combine_catalogs():
    df_uncover = load_uncover_spec()
    df_jades = load_jades_hainline_spec()
    
    combined = pd.concat([df_uncover, df_jades], ignore_index=True)
    
    # Remove duplicates if any (by coordinates)
    # Simple radius match? Or just ID? IDs are from different surveys.
    # Uncover vs JADES fields are different (Abell 2744 vs GOODS-S/N)
    
    print(f"Combined Spectroscopic Catalog: {len(combined)} sources")
    
    # Save
    outfile = DATA_INTERIM / "combined_spectroscopic_catalog.csv"
    combined.to_csv(outfile, index=False)
    print(f"Saved to {outfile}")

if __name__ == "__main__":
    combine_catalogs()

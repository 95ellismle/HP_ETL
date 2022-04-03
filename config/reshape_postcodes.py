import pandas as pd
from pathlib import Path
import shutil


config_dir = Path(__file__).parent
pc_path = config_dir / 'postcodes.parq'
df = pd.read_parquet(pc_path)
df['pc_st'] = df['postcode'].str.slice(0, 1)

write_dir = config_dir / 'postcodes_split'
if write_dir.is_dir():
    shutil.rmtree(write_dir)
write_dir.mkdir()

for key, df in df.groupby('pc_st'):
    df = (df.reset_index(drop=True)
            .drop('pc_st', axis=1)
            .to_parquet(write_dir / f'{key}.parq')
         )
    print("\r"f"Written the {key} file           ")

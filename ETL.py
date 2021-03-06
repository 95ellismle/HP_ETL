import pandas as pd
import pyarrow.feather as ft

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hp_config as hpc

import logging
import json
import subprocess
import sys
import yaml


config_path = hpc.path

# Setup logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

with open('config/categories.json', 'r') as f:
    categories = json.load(f)


def _get_most_common_values(data):
    """Will return a list of all unique elements in data sorted by number of occurances

    Args:
        data: A data structure that can be iterated over (e.g. list or pandas Series or numpy array)
    """
    counts = Counter(data)
    elm_occ = sorted((v, k) for k, v in counts.items())
    return [i[1] for i in elm_occ[::-1]]


pc_dir = Path(__file__).parent / 'config/postcodes_split'
PC_MAP = {f.stem: pd.read_parquet(f) for f in pc_dir.glob('?.parq')}


class ETL:

    _cols = ('id', 'price', 'date_transfer', 'postcode', 'dwelling_type', 'is_new', 'tenure', 'paon', 'saon', 'street', 'locality', 'city', 'district', 'county', 'ppd_cat_type', 'record_amendments')
    _csv_read_dtypes = {'id': 'str',
                   'price': 'int64',
                   'date_transfer': 'datetime',
                   'postcode': 'str',
                   'dwelling_type': 'str',
                   'is_new': 'str',
                   'tenure': 'str',
                   'paon': 'str',
                   'saon': 'str',
                   'street': 'str',
                   'locality': 'str',
                   'city': 'str',
                   'district': 'str',
                   'county': 'str',
                   'ppd_cat_type': 'str',
                   'record_amendments': 'str',
                   }
    with open('config/dtypes.yaml', 'r') as f:
        _df_dtypes = yaml.safe_load(f)
    _raw_dir = Path('./raw')

    _stats_fn = config_path / 'data_stats.yaml'
    _csv_fn = str(_raw_dir / f'pp-$<year>.csv')
    _url_fn = 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-$<year>.csv'

    data = pd.DataFrame({})

    def extract(self, year):
        """Will extract (transform if required) then load the final feather into the data folder

        Transform is part of the csv extract.

        Args:
            year: the year to extract
        """
        self.year = str(year)
        self.csv_fn = Path(self._csv_fn.replace('$<year>', self.year))
        self.url_fn = self._url_fn.replace('$<year>', self.year)

        LOG.debug(f'CSV Name: {self.csv_fn}')
        LOG.debug(f'URL Name: {self.url_fn}')

        # Extract
        LOG.info('Starting Extract')
        self.data = pd.concat([self.data, self._extract(year)])

    def _calc_sort_indices(self, df, cols=['price', 'postcode', 'street',
                                           'city', 'county', 'paon']):
        """Will calculate the sort indices for the columns in a dataframe

        Args:
            df: A dataframe to add columns to
            cols: The cols to add sort indices for
        Returns:
            <pd.DataFrame> df with sorted indices
        """
        df = df.sort_values(['date_transfer', 'county', 'city', 'street'])
        df = df.reset_index(drop=True)
        self._set_sort_index(cols, df)
        for col in cols:
            assert all(sorted(df[col]) == df.iloc[df[f'{col}_sort_index'].values][col])
        return df

    def transform(self):
        """Add some extra columns to improve search speed later"""
        # Combine house num and saon
        self.data = self.data.dropna(axis=0, subset=['price'])
        mask = self.data['saon'] == 'nan'
        self.data.loc[mask, 'saon'] = ''
        self.data.loc[~mask, 'saon'] = ', ' + self.data.loc[~mask, 'saon']
        self.data['paon'] = self.data['paon'] + self.data['saon']
        self.data = self.data.drop('saon', axis=1)

        LOG.info("Setting sort indices")
        self.data = self._calc_sort_indices(self.data)

        # Need to sort postcode for the pc splicing, but need to sort by date -just once for the web app.
        LOG.info('Creating postcode data structure')
        non_na_pc_mask = self.data['postcode'] != 'nan'
        non_na_pc = self.data.loc[non_na_pc_mask, 'postcode'].str.slice(0, 1)
        self._pc_counts = Counter(non_na_pc)
        self.postcode_data = self._postcode_reshape()

        # Create mappings for later
        LOG.info('Mapping streets to a postcode')
        mapping = self._place_to_postcode('street')
        LOG.info("Updating and writing map")
        self._update_json(mapping, config_path / 'maps/street_to_pc.json')

        LOG.info('Mapping cities to a postcode')
        mapping = self._place_to_postcode('city')
        LOG.info("Updating and writing map")
        self._update_json(mapping, config_path / 'maps/city_to_pc.json')

        LOG.info('Mapping counties to a postcode')
        mapping = self._place_to_postcode('county')
        LOG.info("Updating and writing map")
        self._update_json(mapping, config_path / 'maps/county_to_pc.json')

        # Update stats for base.yaml
        LOG.info("Updating data stats")
        self._update_stats_yaml({'max_date': self.data['date_transfer'].iloc[0],
                                 'min_date': self.data['date_transfer'].iloc[-1],
                                 f'len_df_{self.year}': len(self.data),
                                 f'union_poss_postcodes': list(self.postcode_data),
                                 })

    def _write_1_postcode_file(self, pc):
        """Will write a single postcode file"""
        # Create file structure
        dir_name = self._postcode_dir
        dir_name.mkdir(exist_ok=True)

        fn = dir_name / f'{pc[0]}.feather'
        df = self.postcode_data[pc]
        for is_new in (True, False):
            df1 = df[df['is_new'] == is_new]

            for dt in hpc.dwelling_type:
                df2 = df1[df1['dwelling_type'] == dt].drop('is_new', axis=1)

                for tenure in hpc.tenure:
                    new_df = df2[df2['tenure'] == tenure].drop(['tenure', 'dwelling_type'],
                                                               axis=1)
                    if len(new_df):
                        is_new = int(is_new)
                        dwelling_type = hpc.dwelling_type[dt]
                        tenure = hpc.tenure[tenure]
                        fn = dir_name / f'{pc}_IN{is_new}_DT{dwelling_type}_TN{tenure}.feather'

                        new_df = new_df.drop('postcode_sort_index', axis=1)
                        new_df = new_df.sort_values(['date_transfer', 'county', 'city'])
                        new_df = new_df[[i for i in new_df.columns if 'sort_index' not in i]]
                        if len(new_df) > hpc.sort_index_len:
                            new_df = self._calc_sort_indices(new_df)
                        new_df.reset_index(drop=True).to_feather(fn)

    def _set_sort_index(self, col, df):
        """Will set the index to the ordering of the pricing data. This will allow
        an efficient binary search to be used to splice data in requests.

        Will add an extra column, ..._sort_index to the data which contain the column's
        ordering.

        Args:
            col: the column to set the sort index for
            df: the dataframe to change
        """
        if isinstance(col, str):
            col = [col]

        for c in col:
            sort_index_name = f'{c}_sort_index'
            df[sort_index_name] = range(len(df))
            ind = df[[c, sort_index_name]].sort_values(c)[sort_index_name].values
            df.loc[:, sort_index_name] = ind

    def load(self, filename):
        """Will save the dataframe as a feather in the data folder

        Args:
            filename: The filename for the data file that will be saved.
        """
        LOG.info(f"Loading data for {self.year}")
        filename = Path(filename)
        self._data_dir = filename.parent
        self._postcode_dir = self._data_dir / 'postcodes'

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._postcode_dir.mkdir(parents=True, exist_ok=True)

        # Save the postcode data
        LOG.debug("Writing postcode files")
        self._pc_lens = {}
        #for pc in self.postcode_data:
        #    self._write_1_postcode_file(pc)
        with ThreadPoolExecutor(8) as executor:
            postcodes = list(self.postcode_data.keys())
            executor.map(self._write_1_postcode_file, postcodes)

        self.data.to_feather(filename)

    def _update_json(self, data, json_filepath):
        """Will update a json file with the contents of a dictionary

        The dictionary should have the following structure:
            {key1: [val1, val2, ..], key2: [...], ...}

        If json exists will update, if it doesn't will create.

        Args:
            data: a dict with some data to update the json with
            json_filepath: the filepath to the json file to write
        """
        # Read current json file
        curr_data = {}
        if json_filepath.is_file():
            with open(json_filepath, 'r') as f:
                curr_data = json.load(f)
                if curr_data is None:
                    curr_data = {}

        # Update with newer data
        for i in data:
            if i in curr_data:
                curr_data[i] = list(set(curr_data[i] + data[i]))
            else:
                curr_data[i] = data[i]

        # Save the data
        json_filepath.parent.mkdir(exist_ok=True)
        with open(json_filepath, 'w') as f:
            json.dump(curr_data, f)

    def _update_stats_yaml(self, stats):
        """Will update the base.yaml file with stats from the ETL runs

        Will read the current stats yaml and save updated values according to
        some rules (e.g. minimum or maximum).

        Args:
            stats: The stats to save from the ETL runs.
                   For any stats starting with:
                       min_* the minimum value will be saved
                       max_* the maximum value will be saved
        """
        filepath = self._stats_fn
        dt_format = '%Y/%m/%d'
        # Read previous data
        if not filepath.is_file():
            data = {}
        else:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            if data is None: data = {}
            for key in data:
                if key.endswith('_date'):
                    data[key] = pd.to_datetime(data[key], format=dt_format)

        # Merge current and previous
        for key in stats:
            vals = [stats[key]]
            if key in data:
                vals.append(data[key])

            if key.startswith('min_'):
                new_val = min(vals)
            elif key.startswith('max_'):
                new_val = max(vals)
            elif key.startswith('union_'):
                new_val = stats[key]
                key = key[6:]
                if key in data:
                    new_val = list(set(stats[f'union_{key}']).union(data[key]))
            else:
                new_val = stats[key]
            data[key] = new_val

        # Convert timestamps to string
        for key in data:
            val = data[key]
            if isinstance(val, pd._libs.tslibs.timestamps.Timestamp):
                val = val.strftime(dt_format)
            data[key] = val

        # Save merged data
        with open(filepath, 'w') as f:
            yaml.safe_dump(data, f)

    def _postcode_reshape(self):
        """Create individual dfs for each unique beginning pair of letters."""
        d = {}
        c = 0
        df = self.data
        for pc in sorted(self._pc_counts):
            num_vals = self._pc_counts[pc]
            inds = df['postcode_sort_index'].values[c:c+num_vals]
            new_df = df.iloc[inds]
            d[pc] = pd.merge(new_df, PC_MAP[pc], left_on='postcode', right_on='postcode')

            c += self._pc_counts[pc]

        return d

    def _place_to_postcode(self, col):
        """Will create a dictionary mapping every unique occurance of a column to the
           first 2 letters of a postcode.

        This will then be appended to a yaml file in the config directory
        """
        mapping = {}
        for pc in sorted(self.postcode_data):
            new_df = self.postcode_data[pc]

            for val in new_df[col].unique():
                mapping.setdefault(val, []).append(pc)

        return mapping

    def _extract(self, year):
        """Will read the data file from wherever it can be found.

        Will return the dataframe

        Args:
            year: the year that should be extracted
        """
        # First check we have the raw files
        if self.csv_fn.is_file():
            LOG.info("Reading CSV")
            data = self._read_pp_csv(self.csv_fn)

        # If not re-download
        else:
            if not self._raw_dir.is_dir():
                LOG.info("Creating raw directory")
                self._raw_dir.mkdir()
            cmd = ['wget',
                   self.url_fn,
                   '-O', str(self.csv_fn)]
            LOG.info("Downloading file...")
            LOG.debug(f"Command: {' '.join(cmd)}")
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode != 0:
                raise SystemExit(f"Quiting: wget return exit code, {p.returncode}")
            data = self._read_pp_csv(self.csv_fn)

        data = data.drop(['id', 'locality', 'district', 'ppd_cat_type', 'record_amendments'], axis=1)
        return data

    def _read_pp_feather(self, filepaths):
        """Will read the processed feather files.

        Args:
            filepaths: list of filepaths to read
        """
        return pd.concat((pd.read_feather(fp) for fp in filepaths))

    def _read_pp_csv(self, fn):
        """Will read the raw csv file"""
        dtypes = self._csv_read_dtypes
        datetime_cols = [i for i in dtypes if dtypes[i] == 'datetime']
        for col in datetime_cols:
            dtypes.pop(col)

        # Read data
        df = pd.read_csv(fn,
                         names=self._cols,
                         dtype=dtypes,
                         parse_dates=datetime_cols)

        # Add back any cols popped earlier
        for col in datetime_cols:
            dtypes[col] = 'datetime'

        # Sort out strings
        df['postcode'] = df['postcode'].str.upper().str.replace(' ', '')
        for i in ('county', 'street', 'city', 'district',
                  'paon', 'saon', 'locality'):
            df[i] = df[i].fillna('')
            df[i] = df[i].str.lower()
        df['is_new'] = df['is_new'] == 'Y'

        # Humanise the names
        mask = df['tenure'] == 'F'
        df.loc[mask, 'tenure'] = 'Freehold'
        df.loc[~mask, 'tenure'] = 'Leasehold'

        residential = 'Residential'
        df['dwelling_type'] = df['dwelling_type'].map({'F': residential, 'D': residential,
                                                       'S': residential, 'T': residential,
                                                       'O': 'Commercial'})

        # Set the categories to a fixed num
        for cat in categories:
            df[cat] = df[cat].astype(str)
            new_cats = set(df[cat]).difference(categories[cat])
            if new_cats and '' not in new_cats:
                raise SystemExit(f"New categories ({new_cats}) for {cat}, year: {year}")
            df[cat] = pd.Categorical(df[cat], categories=categories[cat])

        # Enforce dtypes
        for col in self._df_dtypes:
            df[col] = df[col].astype(self._df_dtypes[col])

        return df


if __name__ == '__main__':
    curr_year = pd.Timestamp.now().year + 1
    #for year in range(1995, curr_year + 1):

    chunk_size = 1
    for end_y in range(curr_year, 1995, -chunk_size):
        s = end_y - chunk_size
        s = 1995 if s < 1995 else s

        etl = ETL()
        for year in range(s, end_y):
            LOG.info(f'Carrying out year: {year}')
            etl.extract(year)

        etl.transform()
        etl.load(f'data/{year}/pp-{year}.feather')
        del etl


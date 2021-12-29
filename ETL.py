import pandas as pd
import pyarrow.parquet as pq

from collections import Counter
from pathlib import Path

import logging
import subprocess
import sys


# Setup logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)


def _get_most_common_values(data):
    """Will return a list of all unique elements in data sorted by number of occurances

    Args:
        data: A data structure that can be iterated over (e.g. list or pandas Series or numpy array)
    """
    counts = Counter(data)
    elm_occ = sorted((v, k) for k, v in counts.items())
    return [i[1] for i in elm_occ[::-1]]


class ETL:

    _cols = ('id', 'price', 'date_transfer', 'postcode', 'type', 'is_new', 'tenure', 'paon', 'saon', 'street', 'locality', 'city', 'district', 'county', 'ppd_cat_type', 'record_amendments')
    _dtypes = {'id': ('str', 'str'),
               'price': ('int64', 'float64'),
               'date_transfer': ('datetime', 'datetime64[ns]'),
               'postcode': ('str', 'str'),
               'type': ('str', 'str'),
               'is_new': ('str', bool),
               'tenure': ('str', 'str'),
               'paon': ('str', 'str'),
               'saon': ('str', 'str'),
               'street': ('str', 'str'),
               'locality': ('str', 'str'),
               'city': ('str', 'str'),
               'district': ('str', 'str'),
               'county': ('str', 'str'),
               'ppd_cat_type': ('str', 'str'),
               'record_amendments': ('str', 'str'),
               }
    _data_dir = Path('./data')
    _postcode_dir = _data_dir / 'postcodes'
    _raw_dir = Path('./raw')

    _parq_fn = str(_data_dir / f'pp-$<year>.parq')
    _csv_fn = str(_raw_dir / f'pp-$<year>.csv')
    _url_fn = 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-$<year>.csv'
    def __init__(self, year):
        """Will extract (transform if required) then load the final parquet into the data folder

        Transform is part of the csv extract.

        Args:
            year: the year to extract
        """
        self.year = str(year)
        self._parq_fn = Path(self._parq_fn.replace('$<year>', self.year))
        self._csv_fn = Path(self._csv_fn.replace('$<year>', self.year))
        self._url_fn = self._url_fn.replace('$<year>', self.year)
        LOG.debug(f'Parquet Name: {self._parq_fn}')
        LOG.debug(f'CSV Name: {self._csv_fn}')
        LOG.debug(f'URL Name: {self._url_fn}')

        # Extract
        LOG.info('Starting Extract')
        self._extract(year)

        # Transform
        LOG.info('Creating postcode data structure')
        self.postcode_data = self._postcode_reshape()

        # Load
        LOG.info('Writing files')
        self._load(year)

    def _load(self, year):
        """Will save the dataframe as a parquet in the data folder"""
        if not self._data_dir.is_dir():
            LOG.debug("Making data directory")
            self._data_dir.mkdir()
        if not self._postcode_dir.is_dir():
            LOG.debug("Making postcode directory")
            self._postcode_dir.mkdir()

        # Save the postcode data
        LOG.debug("Writing postcode files")
        for i in self.postcode_data:
            fn = self._postcode_dir / f'{i}.parq'
            df = self.postcode_data[i]
            if fn.is_file():
                df = pd.concat([pd.read_parquet(fn), df])
                df = df.sort_values('date_transfer')
            df.to_parquet(fn)

        # Save standard data files
        LOG.debug("Writing data files")
        self.data.to_parquet(self._parq_fn)

    def _postcode_reshape(self):
        """Create individual dfs for each unique beginning pair of letters."""
        # We don't want any Null postcodes
        df = self.data[self.data['postcode'] != 'nan']
        alphabet = _get_most_common_values(df['postcode'].str.slice(0, 1))
        LOG.debug(f"Most common first letters: {alphabet}")

        d = {}
        for lett in alphabet:
            mask = df['postcode'].str.slice(0, 1) == lett
            new_df = df[mask]
            if lett != alphabet[-1]:
                df = df[~mask]

            alphabet2 = _get_most_common_values(new_df['postcode'].str.slice(1, 2))
            for lett2 in alphabet2:
                mask = new_df['postcode'].str.slice(1, 2) == lett2
                d[f'{lett}{lett2}'] = new_df[mask]
                new_df = new_df[~mask]

        return d

    def _extract(self, year):
        """Will read the data file from wherever it can be found.

        Will save the dataframe as self.data

        Args:
            year: the year that should be extracted
        """
        # First check the processed files
        if self._parq_fn.is_file():
            LOG.info("Reading parquet")
            self.data = self._read_pp_parq(self._parq_fn)
        elif self._csv_fn.is_file():
            LOG.info("Reading CSV")
            self.data = self._read_pp_csv(self._csv_fn)
        else:
            if not self._raw_dir.is_dir():
                LOG.info("Creating raw directory")
                self._raw_dir.mkdir()
            cmd = ['wget',
                   self._url_fn,
                   '-O', str(self._csv_fn)]
            LOG.info("Downloading file...")
            LOG.debug(f"Command: {' '.join(cmd)}")
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode != 0:
                raise SystemExit(f"Quiting: wget return exit code, {p.returncode}")
            self.data = self._read_pp_csv(self._csv_fn)

        # Enforce dtypes from the beginning
        for col in self._dtypes:
            dt = self._dtypes[col][1]
            self.data[col] = self.data[col].astype(dt)

    def _read_pp_parq(self, fp):
        """Will read the processed parquet file"""
        return pd.read_parquet(fp)

    def _read_pp_csv(self, fn):
        """Will read the raw csv file"""
        dtypes = {col: self._dtypes[col][0] for col in self._dtypes}
        datetime_cols = [i for i in dtypes if dtypes[i] == 'datetime']
        for col in datetime_cols:
            dtypes.pop(col)

        # Read data
        df = pd.read_csv(fn,
                         names=self._cols,
                         dtype=dtypes,
                         parse_dates=datetime_cols)
        for col in datetime_cols:
            dtypes[col] = 'datetime'

        # Sort out strings
        df['postcode'] = df['postcode'].str.upper().str.replace(' ', '')
        for i in ('county', 'street', 'city', 'district',
                  'paon', 'saon', 'locality'):
            df[i] = df[i].str.title().str.strip()
        df['is_new'] = df['is_new'] == 'Y'

        # Humanise the names
        mask = df['tenure'] == 'F'
        df.loc[mask, 'tenure'] = 'Freehold'
        df.loc[~mask, 'tenure'] = 'Leasehold'

        for curr, new in (('F', 'Flat/Maisonette'),
                          ('D', 'Detached'),
                          ('S', 'Semi-Detached'),
                          ('T', 'Terraced'),
                          ('O', 'Other'),):
            df.loc[df['type'] == curr, 'type'] = new

        # Save prices in thousands
        df['price'] /= 1000

        return df


for year in range(1995, 2022):
    LOG.info(f'Carrying out year: {year}')
    e = ETL(year)


import json
from pathlib import Path
import pandas as pd


cols = ('id', 'price', 'date_transfer', 'postcode', 'dwelling_type', 'is_new',
        'tenure', 'paon', 'saon', 'street', 'locality', 'city', 'district',
        'county', 'ppd_cat_type', 'record_amendments')

# Get unique categories for each column
cats = {}
tenure_map = {'f': 'Freehold', 'l': 'Leasehold', 'u': 'Unknown'}
for i in Path('../raw/').glob('pp*.csv'):
    df = pd.read_csv(i, names=cols)
    for col in ('city', 'county', 'locality', 'tenure', 'district', 'ppd_cat_type', 'record_amendments'):
        for val in df[col].unique():
            if val != val:
                continue
            cats.setdefault(col, set()).add(val.lower())

    for col in ('tenure',):
        for val in df[col].unique():
            if val != val:
                continue
            cats.setdefault(col, set()).add(tenure_map[val.lower()])

    for col in ('ppd_cat_type', 'record_amendments'):
        for val in df[col].unique():
            if val != val:
                continue
            cats.setdefault(col, set()).add(val.upper())

    print(f"{i}, finished")


# Change everything to a list
# It's important everything is sorted
for col in cats:
    cats[col] = sorted(cats[col])


# Write the categories
with open('categories.json', 'w') as f:
    json.dump(cats, f)

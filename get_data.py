import os
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import StratifiedKFold , train_test_split
from resources.data_io import save_data, save_mappings
from resources.spans import SpanCollection

# Settings:
K =         5   # Number of cross validation splits
SEED =      42  # Random seed
DATAURL =   'https://zenodo.org/records/10891602/files/food_recall_incidents.csv?download=1'
DATASET =   'incidents'
DATA_DIR =  f'data/{DATASET}/'
SPLIT_DIR = f'{DATA_DIR}splits/'

# Download Data:
data_path, headers = urlretrieve(DATAURL, f'{DATA_DIR}{DATASET}.csv')
for name, value in headers.items():
    print(name, value)

# Load Data:
## Load "incidents":
data = pd.read_csv(data_path, index_col=0)

# fill nan-values:
data['product']          = data['product'].fillna('')
data['product-category'] = data['product-category'].fillna('')

data['hazard']           = data['hazard'].fillna('')
data['hazard-category']  = data['hazard-category'].fillna('')

data['country']          = data['country'].fillna('na')
data['language']         = data['language'].fillna('na')

# parse spans:
for col in ['product-title', 'product-text', 'hazard-title', 'hazard-text', 'supplier-title', 'supplier-text']: 
    if col in data.columns: data[col] = [SpanCollection.parse(item) for item in data[col].fillna('')]
    else:                   data[col] = [SpanCollection()] * len(data)

print(f"N = {len(data):d}")

# Vectorize labels:
unique_labels = {}

for column in ['product', 'hazard', 'product-category', 'hazard-category']:
    # extract and sort unique values:
    unique_labels[column] = np.unique(data[column].values)
    unique_labels[column].sort()

    # create label-to-integer mapping:
    label2index = {l:i for i,l in enumerate(unique_labels[column])}

    # replace strings with integers:
    data[column] = data[column].apply(lambda label: label2index[label])

# save mappings:
os.makedirs(SPLIT_DIR, exist_ok=True)

save_mappings(SPLIT_DIR,
    unique_labels['product'],
    unique_labels['hazard'],
    unique_labels['product-category'],
    unique_labels['hazard-category']
)

# Create K-Fold splits
def filter_by_support(rows, column, min_support):
    for i_label, label in enumerate(unique_labels[column]):
        mask = (data.loc[rows, column].values == i_label)

        if sum(mask) < min_support:
            rows = rows[~mask]
            print(f'{column.upper()}: dropped class "{label}" with n_samples = {sum(mask)} < {min_support:d}.')

    return rows

for label_fine, label_coarse, columns in [
        ('product', 'product-category', [col for col in data.columns if not col.startswith('hazard')]),
        ('hazard',  'hazard-category',  [col for col in data.columns if not col.startswith('product')])
    ]:

    # drop coarse labels with less than K samples:
    i_filtered = filter_by_support(data.index, label_coarse, K)

    # create K-fold splits:
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    for split, (i_train, i_test) in enumerate(kf.split(i_filtered, data.loc[i_filtered, label_coarse].values)):
        i_train = filter_by_support(i_filtered[i_train], label_coarse, 2)
        i_test  = i_filtered[i_test]
        # split off validation set:
        i_train, i_valid = train_test_split(
            i_train,
            test_size=.1,
            stratify=data.loc[i_train, label_coarse].values,
            shuffle=True,
            random_state=SEED
        )

        # save mappings:
        save_data(SPLIT_DIR, split, label_fine,
            data.loc[i_train, columns],
            data.loc[i_valid, columns],
            data.loc[i_test,  columns]
        )

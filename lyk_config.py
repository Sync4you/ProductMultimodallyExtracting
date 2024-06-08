
k = 50
conf_th = 0.7

import pandas as pd
from pathlib import Path


def load_data():
    nrows = 55
    df = pd.read_csv('../data.csv',
                     nrows=nrows, usecols=['posting_id', 'image', 'title'])
    img_dir = Path('../../images/')
    return df, img_dir

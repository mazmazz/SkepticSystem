import numpy as np
import pandas as pd
from csv_prices import load_csv_prices

# parent submodules
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
#
from preprocessors.delta_transformer import DeltaTransformer
#
sys.path.pop(0)
# end parent submodules

def load_prices(
    instrument
    , granularity
    , start_date=None
    , end_date=None 
    , source='csv'
    , **kwargs
):
    if source == 'csv':
        return load_csv_prices(instrument, granularity, start_date, end_date, **kwargs)
    else:
        return None

def get_target(
    prices
    , end_offset
    , start_offset=-1
    , column='open'
    , operation='direction'
):
    dtr = DeltaTransformer(start_offset, end_offset, step=0, operation=operation, remainder=True)
    price_col = prices if len(prices.shape) < 2 else prices[column]
    output = dtr.transform(price_col)
    if isinstance(output, pd.Series):
        output.rename('target', inplace=True)
    return output

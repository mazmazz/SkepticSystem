import numpy as np
import pandas as pd
from .csv_prices import load_csv_prices

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessors.delta_transformer import DeltaTransformer
sys.path.pop(0)
# end parent submodules

def load_prices(
    instrument
    , granularity
    , start_index=None
    , end_index=None 
    , source='csv'
    , **kwargs
):
    if source == 'csv':
        return load_csv_prices(instrument, granularity, start_index, end_index, **kwargs)
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

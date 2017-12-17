import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import make_union

# parent submodules
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
#
from datasets import load_prices, get_target
#
sys.path.pop(0)
# end parent submodules

def do_candidate(params):
    # load prices
    prices, target = do_data(**params['data__params'])
    prices_trade = prices.copy()

    # do indicators
    

def do_data(
    instrument
    , granularity
    , end_target
    , source='csv'
    , start_index=None
    , end_index=None
    , sample_len=None
    , dir='.'
):
    prices = load_prices(instrument, granularity, start_index=start_index, end_index=end_index, source=source, sample_len=sample_len, dir=dir)
    target = get_target(prices, end_target)
    return prices, target

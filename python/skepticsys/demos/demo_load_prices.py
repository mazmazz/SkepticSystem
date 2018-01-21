import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import make_union

# parent submodules
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
#
from datasets.load_prices import load_prices, get_target
#
sys.path.pop(0)
# end parent submodules

prices = load_prices('USDJPY','H1',source='csv',dir='D:\\Projects\\Prices\\',sample_len=-12000)
target = get_target(prices, -61)

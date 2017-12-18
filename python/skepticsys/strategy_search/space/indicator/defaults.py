import hyperopt as hp

# parent imports
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) # 3 levels up
from preprocessors.copy_transformer import CopyTransformer
from utils import _get_price_field
sys.path.pop(0)
# end parent imports

def copy_field(name, prices, input_names=None, **kwargs):
    if not isinstance(input_names, list):
        input_names = [input_names]

    cpt = CopyTransformer()
    output = []
    for input_name in input_names:
        output.append(cpt.transform(_get_price_field(prices, input_name)))
    
    if len(output) == 1:
        return output[0]
    elif isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series):
        return pd.concat(output, axis=1)
    else:
        return np.concatenate(output, axis=1)

default_indicators = {
    'copy_open': {'callable_': copy_field, 'input_': 'open'},
    'copy_high': {'callable_': copy_field, 'input_': 'high'},
    'copy_low': {'callable_': copy_field, 'input_': 'low'},
    'copy_close': {'callable_': copy_field, 'input_': 'close'},
    'copy_volume': {'callable_': copy_field, 'input_': 'volume'},
    'ADX': {'callable_':'talib', 'timeperiod': 14},
    'AROONOSC': {'callable_':'talib', 'timeperiod': 14},
    'BOP': {'callable_':'talib'},
    'CCI': {'callable_':'talib', 'timeperiod': 14},
    'CMO': {'callable_':'talib', 'timeperiod': 14},
    'MACD': {'callable_':'talib', 'fastperiod': 12, 'signalperiod': 9, 'slowperiod': 26},
    'MFI': {'callable_':'talib', 'timeperiod': 14},
    'MINUS_DI': {'callable_':'talib', 'timeperiod': 14},
    'MOM': {'callable_':'talib', 'timeperiod': 10},
    'PLUS_DI': {'callable_':'talib', 'timeperiod': 14},
    'ROCP': {'callable_':'talib', 'timeperiod': 10},
    'RSI': {'callable_':'talib', 'timeperiod': 14},
    'STOCH': {'callable_':'talib', 'fastk_period': 5,
        'slowd_matype': 0,
        'slowd_period': 3,
        'slowk_matype': 0,
        'slowk_period': 3},
    'STOCHRSI': {'callable_':'talib', 'fastd_matype': 0,
        'fastd_period': 3,
        'fastk_period': 5,
        'timeperiod': 14},
    'ULTOSC': {'callable_':'talib', 'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
    'AD': {'callable_':'talib'},
    'ADOSC': {'callable_':'talib', 'fastperiod': 3, 'slowperiod': 10},
    'OBV': {'callable_':'talib'},
    'NATR': {'callable_':'talib', 'timeperiod': 14},
    'STDDEV': {'callable_':'talib', 'nbdev': 1, 'timeperiod': 5},
    'cvi': {'callable_':'tulipy', 
        'period': 10},
    'emv': {'callable_':'tulipy'},
    'fosc': {'callable_':'tulipy', 
        'input_': {'real':'close'},
        'period': 25},
    'kvo': {'callable_':'tulipy', 
        'long_period': 55, 'short_period': 34},
    'mass': {'callable_':'tulipy', 
        'period': 25},
    'ppo': {'callable_':'tulipy',
        'input_': {'real':'close'},
        'long_period': 26, 'short_period': 12},
    'qstick': {'callable_':'tulipy', 
        'period': 25},
    'volatility': {'callable_':'tulipy', 
        'input_': {'real':'close'},
        'period': 11},
    'nvi': {'callable_':'tulipy'},
    'pvi': {'callable_':'tulipy'}
}

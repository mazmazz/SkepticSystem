from hyperopt import hp
from hyperopt.pyll.base import scope
from talib import MA_Type

# parent imports
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # 2 levels up
from preprocessors.copy_transformer import CopyTransformer
from utils import _get_price_field
sys.path.pop(0)
# end parent imports

def get_indicator_space(
    indi_settings={}
):
    space = {}

    # modify default indicators with input settings
    indicators = {
        **default_indicators
        , **indi_settings # overwrites above keys if they exist
    }

    subindi_names = ('_indi', '_ma_pre', '_ma_post')
    for indi in indicators:
        if not bool(indicators[indi]): # evals True for {}, False for None or False
            continue

        subindi_settings = {name: indicators[indi].pop(name, None) for name in subindi_names if name in indicators[indi]}
        indi_space = {
            '_indi': get_subindi_space(indi, '__indi', delta_ma=True) if '_indi' not in subindi_settings else subindi_settings['_indi']
            , '_ma_pre': get_subindi_space(indi, '__ma_pre', ma=True, ma_pre=True) if '_ma_pre' not in subindi_settings else subindi_settings['_ma_pre']
            , '_ma_post': get_subindi_space(indi, '__ma_post', ma=True) if '_ma_post' not in subindi_settings else subindi_settings['_ma_post']
            , '_params': indicators[indi]
        }

        space[indi] = hp.choice('indicators__'+indi, [None, indi_space])
    
    return {'indicator__params': space}

def get_subindi_space(indi, prefix, ma=False, delta=True, shift=True, ma_pre=False, delta_ma=False):
    space = {
        **({'_delta': get_delta_space(indi, prefix, ma=delta_ma)} if delta else {})
        , **({'_shift': get_shift_space(indi, prefix)} if shift else {})
        , **({'_ma': get_ma_params(indi, prefix, pre=ma_pre)} if ma else {})
    }

    return hp.choice(indi+prefix, [None, space])

def get_ma_params(indi, prefix, pre=False):
    return {
        'timeperiod': 30
        , 'matype': MA_Type.EMA
        , '_pre': pre
        , 'inputs_': True
    }

def get_delta_space(indi, prefix, ma=False):
    space = {
        '_base': hp.choice(indi+prefix+'__delta__base', [False,True])
        , '_diff': hp.choice(indi+prefix+'__delta__diff', [None, {
            '_ma': hp.choice(indi+prefix+'__delta__diff__ma', [None, get_ma_params(indi, prefix+'__delta__diff') if ma else None])
            , '_shift': hp.choice(indi+prefix+'__delta__diff__shift', [None, get_shift_space(indi, prefix+'__delta__diff')])
            , 'operation': 'diff'
            , 'start': 0
            , 'stop': scope.int(hp.quniform(indi+prefix+'__delta__diff__stop', 2, 15, 1))
            , 'step': scope.int(hp.quniform(indi+prefix+'__delta__diff__step', 1, 2, 1))
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__diff__fixed_start', [False,True])
        }])
        , '_percent': hp.choice(indi+prefix+'__delta__percent', [None, {
            '_ma': hp.choice(indi+prefix+'__delta__percent__ma', [None, get_ma_params(indi, prefix+'__delta__percent') if ma else None])
            , '_shift': hp.choice(indi+prefix+'__delta__percent__shift', [None, get_shift_space(indi, prefix+'__delta__percent')])
            , 'operation': 'percent'
            , 'start': 0
            , 'stop': scope.int(hp.quniform(indi+prefix+'__delta__percent__stop', 2, 15, 1))
            , 'step': scope.int(hp.quniform(indi+prefix+'__delta__percent__step', 1, 2, 1))
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__percent__fixed_start', [False,True])
        }])
        , '_direction': hp.choice(indi+prefix+'__delta__direction', [None, {
            '_ma': hp.choice(indi+prefix+'__delta__direction__ma', [None, get_ma_params(indi, prefix+'__delta__direction') if ma else None])
            , '_shift': hp.choice(indi+prefix+'__delta__direction__shift', [None, get_shift_space(indi, prefix+'__delta__direction')])
            , 'operation': 'direction'
            , 'start': 0
            , 'stop': scope.int(hp.quniform(indi+prefix+'__delta__direction__stop', 2, 15, 1))
            , 'step': scope.int(hp.quniform(indi+prefix+'__delta__direction__step', 1, 2, 1))
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__direction__fixed_start', [False,True])
        }])
    }

    return hp.choice(indi+prefix+'__delta', [None, space])

def get_shift_space(indi, prefix):
    space = {
        'start': 1
        , 'stop': scope.int(hp.quniform(indi+prefix+'__shift__stop', 2, 15, 1))
        , 'step': 1 # scope.int(hp.quniform(indi+prefix+'__shift__step', 1, 5, 1))
    }
    return hp.choice(indi+prefix+'__shift__params', [None, space])

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

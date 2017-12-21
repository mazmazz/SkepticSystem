from hyperopt import hp
from talib import MA_Type
from .defaults import default_indicators

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
            , 'stop': hp.quniform(indi+prefix+'__delta__diff__stop', 2, 15, 1)
            , 'step': hp.quniform(indi+prefix+'__delta__diff__step', 1, 2, 1)
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__diff__fixed_start', [False,True])
        }])
        , '_percent': hp.choice(indi+prefix+'__delta__percent', [None, {
            '_ma': hp.choice(indi+prefix+'__delta__percent__ma', [None, get_ma_params(indi, prefix+'__delta__percent') if ma else None])
            , '_shift': hp.choice(indi+prefix+'__delta__percent__shift', [None, get_shift_space(indi, prefix+'__delta__percent')])
            , 'operation': 'percent'
            , 'start': 0
            , 'stop': hp.quniform(indi+prefix+'__delta__percent__stop', 2, 15, 1)
            , 'step': hp.quniform(indi+prefix+'__delta__percent__step', 1, 2, 1)
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__percent__fixed_start', [False,True])
        }])
        , '_direction': hp.choice(indi+prefix+'__delta__direction', [None, {
            '_ma': hp.choice(indi+prefix+'__delta__direction__ma', [None, get_ma_params(indi, prefix+'__delta__direction') if ma else None])
            , '_shift': hp.choice(indi+prefix+'__delta__direction__shift', [None, get_shift_space(indi, prefix+'__delta__direction')])
            , 'operation': 'direction'
            , 'start': 0
            , 'stop': hp.quniform(indi+prefix+'__delta__direction__stop', 2, 15, 1)
            , 'step': hp.quniform(indi+prefix+'__delta__direction__step', 1, 2, 1)
            , 'fixed_start': True # hp.choice(indi+prefix+'__delta__direction__fixed_start', [False,True])
        }])
    }

    return hp.choice(indi+prefix+'__delta', [None, space])

def get_shift_space(indi, prefix):
    space = {
        'start': 1
        , 'stop': hp.quniform(indi+prefix+'__shift__stop', 2, 15, 1)
        , 'step': 1 # hp.quniform(indi+prefix+'__shift__step', 1, 5, 1)
    }
    return hp.choice(indi+prefix+'__shift__params', [None, space])

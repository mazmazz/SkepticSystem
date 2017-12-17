from hyperopt import hp
from talib import MA_Type

def get_indicator_space():
    return {}
    space = {
        'stoch': hp.choice('indicator__stoch', [
            None
            , {
                '_prema': hp.choice('stoch__prema', [
                    None, 30 # hp.x('stoch__prema__period', 30)
                ])
                , '_postma': hp.choice('stoch__postma', [
                    None, 30 # hp.x('stoch__postma__period', 30)
                ])
                , '_postdelta': hp.choice('stoch__postdelta', [
                    None, {
                        'type': 'change'
                        # ???
                    }
                ])
                , '_multi': {
                    ['slowk_matype','slowd_matype']: hp.choice('stoch__matype_', [MA_Type.SMA, MA_Type.EMA])
                }
                , 'callable_': 'talib'
                , 'fastk_period': 5
                , 'slowk_period': 3
                , 'slowd_period': 3
            }
        ])
    }
    
    return space

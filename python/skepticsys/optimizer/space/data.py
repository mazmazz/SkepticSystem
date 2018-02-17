from hyperopt import hp

def get_data_space(
    instruments=['USDJPY','USDCAD','EURUSD']
    , granularities=['M5','M15','M30','H1','H2','H4','H8','H12','D1']
    , source='csv'
    , start_index = None
    , end_index = None
    , dir = '.'
    , end_target = None
    , start_buffer = 1000
    , end_buffer = 0
):
    space = {
        'instrument': hp.choice('data__instrument', instruments)
        , 'granularity': hp.choice('data__granularity', granularities)
        , 'source': source
        , 'start_index': start_index
        , 'end_index': end_index
        , 'dir': dir
        , 'start_buffer': 1000
        , 'end_buffer': 0
    }

    if end_target is not None:
        if isinstance(end_target, list):
            space['end_target'] = hp.choice('data__end_target', end_target)
        else:
            space['end_target'] = end_target
    else:
        space['end_target'] = hp.quniform('data__end_target', -61, -20, 1)

    space['start_target'] = -1 # todo: expose as setting, hardcode this for now

    return {
        'data__params': space
    }

from hyperopt import hp

def get_data_space(
    instruments=['USDJPY','USDCAD','EURUSD']
    , granularities=['M5','M15','M30','H1','H2','H4','H8','H12','D1']
    , source='csv'
    , start_index = None
    , end_index = None
    , dir = '.'
    , start_target = -1
    , end_target = None
    , start_buffer = 1000
    , end_buffer = 0
    , instrument=None # to define explicitly
    , granularity=None  # to define explicitly
    , datetime_convert=False
    , datetime_format='%Y%m%d%H%M'
):
    space = {
        'instrument': instrument or (hp.choice('data__instrument', instruments) if isinstance(instruments, list) else instruments)
        , 'granularity': granularity or (hp.choice('data__granularity', granularities) if isinstance(granularities, list) else granularities)
        , 'source': source
        , 'start_index': start_index
        , 'end_index': end_index
        , 'dir': dir
        , 'start_buffer': start_buffer
        , 'end_buffer': end_buffer
        , 'start_target': start_target
        , 'index_to_datetime': datetime_convert
        , 'datetime_format': datetime_format
    }

    if end_target is not None:
        if isinstance(end_target, list):
            space['end_target'] = hp.choice('data__end_target', end_target)
        else:
            space['end_target'] = end_target
    else:
        space['end_target'] = hp.quniform('data__end_target', -61, -20, 1)

    return {
        'data__params': space
    }

from hyperopt import hp

def get_data_space(
    instruments=['USDJPY','USDCAD','EURUSD']
    , granularities=['M5','M15','M30','H1','H2','H4','H8','H12','D1']
    , source='csv'
    , start_index = None
    , end_index = None
    , sample_len = None
    , dir = '.'
    , end_target = -2
):
    return {
        'data__params': {
            'instrument': hp.choice('data__instrument', instruments)
            , 'granularity': hp.choice('data__granularity', granularities)
            , 'end_target': end_target
            , 'source': source
            , 'start_index': start_index
            , 'end_index': end_index
            , 'sample_len': sample_len
            , 'dir': dir
        }
    }

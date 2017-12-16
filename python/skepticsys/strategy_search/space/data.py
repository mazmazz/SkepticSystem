from hyperopt import hp

def get_data_space(
    instruments=['USDJPY','USDCAD','EURUSD']
    , granularities=['M5','M15','M30','H1','H2','H4','H8','H12','D1']
    , source='oanda'
    , start_index = '20110101'
    , end_index = None
):
    return {
        'data__params': {
            'instrument': hp.choice('data__instrument', instruments)
            , 'granularity': hp.choice('data__granularity', granularities)
            , 'source': source
            , 'start_date': start_index
            , 'end_date': end_index
        }
    }

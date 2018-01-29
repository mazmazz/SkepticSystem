import os
import pandas as pd
import datetime
import hashlib

cache_ = {} # params_, hash_, df_

def file_hash(filename):
    h = hashlib.sha256()
    with open(filename, 'rb', buffering=0) as f:
        for b in iter(lambda : f.read(128*1024), b''):
            h.update(b)
    return h.hexdigest()

def check_hash(fn, base_hash):
    if os.path.isfile(fn):
        compare_hash = file_hash(fn)
        return base_hash == compare_hash
    else:
        return True # in case file is unavailable or deleted, assume it's intermittent

def load_csv_prices(
    instrument
    , granularity
    , start_index=None
    , end_index=None
    , sample_len=None
    , dir='.'
    , index_col=0
    , header_row=None
    , index_to_datetime=False
    , from_test=False
):
    params_dict = locals()
    params = ', '.join(['{}={}'.format(k, params_dict[k]) for k in sorted(list(params_dict))])
    
    # load data
    fn = os.path.join(dir, '%s_%s.csv'%(instrument, granularity))

    # check cache
    cache_key = (fn,params)
    if (
        cache_key in cache_ 
        and check_hash(fn, cache_[cache_key]['hash_'])
    ):
        #print('Loading from cache')
        return cache_[cache_key]['data_']
    else:
        data = pd.read_csv(fn, index_col=index_col, header=header_row)

    # truncate samples only if neither start/end_index are datetimes
    if not isinstance(start_index, datetime.datetime) and not isinstance(end_index, datetime.datetime):
        data = truncate_df(data, start_index=start_index, end_index=end_index, sample_len=sample_len, from_test=from_test)

    # rename columns
    col_names = (
        ['open','high','low','close','volume'] if data.shape[1] <= 5
        else ['open','high','low','close','adjclose','volume']
    )
    col_names = col_names[:min(len(data.columns), len(col_names))]
    data.columns = col_names
    data.index.name = 'date'

    # make index datetime
    if index_to_datetime:
        data = make_datetimeindex(data)

    # if start/end_index are datetime, now truncate samples
    if isinstance(start_index, datetime.datetime) or isinstance(end_index, datetime.datetime):
        data = truncate_df(data, start_index=start_index, end_index=end_index, sample_len=sample_len, from_test=from_test)

    # store cache
    cache_[cache_key] = {
        'hash_': file_hash(fn)
        , 'data_': data
    }

    return data

def truncate_df(data, start_index=None, end_index=None, sample_len=None, from_test=False, start_buffer=1000):
    """
    Parameters
    ----------
    start_index: 
        Index value to lookup, inclusive

    end_index: 
        Index value to lookup, non-inclusive.

    sample_len: int or dict {'train','test','target'}
        If int, grab at least this many rows of data. Negative int means calculate bounds from
        end location.
        
        If dict and `from_test` is True, row lookup will generate bounds from train, test, and target. 
        If `from_test` is False, only the total of train, test, and target is considered. Target must include
        both the start and end offsets. All numbers must be positive.

    from_test: bool, default False
        Interpret start and end indexes as the test bounds, rather than the total data bounds.

    start_buffer: int, default 1000
        Extra data to retrieve prior to starting bounds. Only used if `from_test` is True.
    """
    if start_index is not None:
        start_loc = data.index.get_loc(start_index)
    else:
        start_loc = 0
    
    if end_index is not None:
        end_loc = data.index.get_loc(end_index)
    else:
        end_loc = len(data)

    if isinstance(sample_len, dict):
        train_len = int(abs(sample_len['train']))
        test_len = int(abs(sample_len['test']))
        target_len = int(abs(sample_len['target']))
        sample_len = int(sum([train_len, test_len, target_len]))
    else:
        train_len, test_len, target_len = None, None, None

    if from_test:
        if sample_len is not None:
            if start_index is not None:
                if train_len is None:
                    raise ValueError('Train length must be specified separately if start_index is specified.')
                query_loc = start_loc - train_len - start_buffer
                if query_loc < 0:
                    raise ValueError('Not enough data for requested train_len %s and start_index' % (train_len, start_index))
            else:
                query_loc = end_loc - sample_len - start_buffer
                if query_loc < start_loc:
                    raise ValueError('Not enough data for requested sample_len %s' % sample_len)
            start_loc = query_loc
        
        if target_len is not None and end_index is not None: # if end_index is not specified, we don't care about the end bounds.
            query_loc = end_loc + target_len
            if query_loc > len(data):
                raise ValueError('Not enough data for requested target_len' % (target_len))
            end_loc = query_loc
    else:
        if sample_len is not None and sample_len != 0:
            if sample_len > 0:
                end_loc = start_loc+sample_len
            else: 
                start_loc = end_loc-sample_len

    output = data[start_loc:end_loc]        

    return output

def make_datetimeindex(df):
    # first convert index to string type (dtype object)
    if df.index.dtype != object:
        df_index = df.index.map(str)
    else:
        df_index = df.index

    # convert index to datetime
    df.index = pd.to_datetime(df_index)
    return df

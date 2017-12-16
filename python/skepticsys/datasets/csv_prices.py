import os
import pandas as pd
import datetime

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
):
    # load data
    fn = os.path.join(dir, '%s_%s.csv'%(instrument, granularity))
    data = pd.read_csv(fn, index_col=index_col, header=header_row)

    # truncate samples only if neither start/end_index are datetimes
    if not isinstance(start_index, datetime.datetime) and not isinstance(end_index, datetime.datetime):
        data = truncate_df(data, start_index=start_index, end_index=end_index, sample_len=sample_len)

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
        data = truncate_df(data, start_index=start_index, end_index=end_index, sample_len=sample_len)

    return data

def truncate_df(data, start_index=None, end_index=None, sample_len=None):
    if start_index is not None:
        start_loc = data.index.get_loc(start_index)
    else:
        start_loc = 0
    
    if end_index is not None:
        end_loc = data.index.get_loc(end_index)
    else:
        end_loc = len(data)

    output = data[start_loc:end_loc]

    if sample_len is not None and sample_len != 0:
        if sample_len > 0: 
            output = output[0:sample_len]
        else: 
            output = output[sample_len:len(output)]

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

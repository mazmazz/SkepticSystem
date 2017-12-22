
import numpy as np
import pandas as pd

def _gen_ranges(start, stop, step, fixed_start=False, remainder=True):
        # detect reverse range, and enforce step sign
        is_reverse = stop < start
        
        # fill in step if 0
        if step is None or step == 0: 
            step = (stop-abs(start)) if not is_reverse else -(stop-abs(start))

        if (is_reverse and step > 0) or (not is_reverse and step < 0): 
            step = -step
        
        # https://stackoverflow.com/questions/14048728/generate-list-of-range-tuples-with-given-boundaries-in-python
        current = next_current = start
        while (next_current < stop) if not is_reverse else (next_current > stop):
            next_current = next_current + step
            if (next_current < stop) if not is_reverse else (next_current > stop):
                yield (current, next_current)
            elif remainder: # elif remainder and stop-1 > next_current-step:
                yield (current, stop) # yield (current, stop-1)
            else:
                break
            if not fixed_start: current = next_current

def _rename_pandas(arr, name, append='after', inplace=False):
    def do_append(root, name, append):
        append = append.lower()
        if append in ['after','post']:
            return root+name
        elif append in ['before','pre']:
            return name+root
        else:
            return name

    if not inplace:
        arr = arr.copy()

    if len(arr.shape) > 1:
        arr.columns = [do_append(str(col), name, append) for col in list(arr.columns)]
    else:
        arr.rename(do_append((arr.name or ''), name, append))

    if not inplace:
        return arr

def _get_price_field(prices, field, input_names={}):
    is_pandas = isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series)
    col_names = (
        ['open','high','low','close','volume'] if prices.shape[1] <= 5
        else ['open','high','low','close','adjclose','volume']
    )
    if field in input_names:
        field = input_names[field]

    if is_pandas:
        if field in prices:
            output = prices[field]
        else:
            # if field not in col_names:
            #     print('DEBUG: field not in col_names')
            #     import pdb; pdb.set_trace()
            output = prices.iloc[:,col_names.index(field)]
    else:
        output = prices[:,col_names.index(field)]
    
    return output

def arr_to_datetime(y_pred, y_true=None):
    """Convert pred to Pandas Series with DatetimeIndex.

    Parameters
    ----------
    y_pred: array-like, shape (n_samples,)
        Array to convert

    y_true: array-like, shape (n_samples,), optional
        If X is passed and y_pred is a numpy array, y_pred will be
        converted to Pandas Series with X index.

    Returns
    -------
    y_pred Pandas Series, shape (n_samples,), with DatetimeIndex
    """
    if (
        (isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series))
        and isinstance(y_pred.index, pd.DatetimeIndex)
    ):
        return y_pred

    # convert y_pred to Pandas series
    if isinstance(y_pred, np.ndarray):
        if not isinstance(y_true, pd.Series) and not isinstance(y_true, pd.DataFrame):
            raise ValueError('y_pred is a Numpy array and y_true ({}) must be a Pandas Series.'.format(type(y_true)))
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] <= 1:
                y_pred = y_pred.reshape(-1,)
            # else:
            #     raise ValueError('y_pred must be one-dimensional or have only one column. Has %s' % (y_pred.shape[1]))
        y_pred = pd.Series(y_pred, index=y_true.index)
    elif isinstance(y_pred, pd.DataFrame) and len(y_pred.columns) == 1:
        # if len(y_pred.columns) != 1:
        #     raise ValueError('y_pred must have one column. Has %s' % (len(y_pred.columns)))
        y_pred = y_pred.iloc[:,0]

    # convert y_pred index to DatetimeIndex
    # first convert index to string type (dtype object)
    if y_pred.index.dtype != object:
        y_pred_index = y_pred.index.map(str)
    else:
        y_pred_index = y_pred.index

    # convert index to datetime
    y_pred.index = pd.to_datetime(y_pred_index)
    
    return y_pred

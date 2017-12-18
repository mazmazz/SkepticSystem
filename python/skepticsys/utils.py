
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
            if field not in col_names:
                print('DEBUG: field not in col_names')
                import pdb; pdb.set_trace()
            output = prices.iloc[:,col_names.index(field)]
    else:
        output = prices[:,col_names.index(field)]
    
    return output

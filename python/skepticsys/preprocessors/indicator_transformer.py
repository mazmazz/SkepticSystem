import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _pprint
import logging
import inspect

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _get_price_field
sys.path.pop(0)
# end parent submodules

def _is_indicator_talib(name):
    try:
        import talib
    except ImportError:
        logging.getLogger(__name__).debug('talib failed to import or does not exist, continuing')
        return False
    return name.upper() in talib.get_functions()

def _is_indicator_tulip(name):
    try:
        import tulipy
    except ImportError:
        logging.getLogger(__name__).debug('tulipy failed to import or does not exist, continuing')
        return False
    return hasattr(tulipy, name.lower())

def _get_indicator_callable(name, src=None, default_lib='talib'):
    if callable(src):
        return src
    else:
        if isinstance(src, str):
            checks = [src]
        else:
            libs = ['talib','tulip']
            if not isinstance(default_lib, list):
                default_lib = [default_lib]
            checks = [*default_lib, *[lib for lib in libs if lib not in default_lib]]

        for check in checks:
            if check.lower() == 'talib' and _is_indicator_talib(name):
                return _talib_indicator
            elif check.lower() in ['tulip','tulipy'] and _is_indicator_tulip(name):
                return _tulip_indicator
        
        return None

def _get_indi_name(key):
    if not '__' in key:
        return key
    else:
        return key.split('__')[0]

def _name_prices(prices):
    is_pandas = isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series)
    prices = prices.copy() if is_pandas else np.copy(prices)

    if is_pandas:
        try:
            prices.columns = prices.columns.str.lower()
        except:
            # will fail if column index is not string, in which case we'll try lookup by numeric index
            pass
    # else do nothing, we rely on numeric index for numpy
    
    return prices

def _format_indicator_output(data, is_pandas):
    if is_pandas:
        data = pd.DataFrame(data)
    else:
        data = np.stack(list(data.values())).transpose()
    return data

def _get_input_names(inputs, indi_fields, input_df):
    # expected: [[input, input], [input, input]], in order of field definition
    # or [{field: input, ...},{field: input, ...}]
    # allowed: True - only applies for single fields; False - returns empty indi input
    # allowed: [input, input, input] - makes sense for single fields, otherwise will output [{field1: input},{field2: input}] etc
    # allowed: empty inputs [], assign default inputs

    if isinstance(inputs, dict):
        inputs = [inputs]

    if inputs is None or inputs is False or len(indi_fields) <= 0:
        return [{}]
    elif inputs is True:
        return [{field_name: col_name for field_name in indi_fields} for col_name in input_df]
    elif len(inputs) <= 0:
        return [{field_name: field_name for field_name in indi_fields}] # if field_name in input_df}]
    else:
        output = []

        for inp in inputs:
            if isinstance(inp, list):
                output.append({field_name: inp_name for field_name, inp_name in zip(indi_fields, inp)})
            elif isinstance(inp, dict):
                output.append({inp_field: inp[inp_field] for inp_field in inp})
            else:
                output.append({field_name: inp for field_name in indi_fields})
        
        return output

########## talib ##########

def _talib_indicator(name, prices, input_names=None, **kwargs):
    # optional import
    try:
        import talib
    except ImportError:
        logging.getLogger(__name__).debug('talib failed to import or does not exist, continuing')
        return None

    # setup
    is_pandas = isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series)
    prices = _name_prices(prices)
    base_name = _get_indi_name(name)
    indi_function = talib.abstract.Function(base_name.upper())
    indi_function.parameters = kwargs
    
    # get input fields
    # Most indis have 'price' if they take only one field, or 'prices' if they take multiple
    # OBV has both 'price' (close) and 'prices' ([volume]) for some reason, so include both in indi_fields
    indi_fields_in = indi_function.input_names
    indi_fields = []
    if 'price' in indi_fields_in: 
        indi_fields.append(indi_fields_in['price'])
    if 'prices' in indi_fields_in: 
        indi_fields.extend(indi_fields_in['prices'])

    # get inputs
    input_names = _get_input_names(input_names, indi_fields, prices)
    output = {}
    for i, names in enumerate(input_names):
        indi_inputs = {
            field: (
                _get_price_field(prices, field, input_names=names).as_matrix().astype(np.float64) if is_pandas
                else _get_price_field(prices, field, input_names=names).astype(np.float64)
            ) for field in indi_fields
        }

        # get result
        result = indi_function.run(indi_inputs)

        # format output
        for j, out_name in enumerate(indi_function.output_names):
            set_name = ':'.join('{}'.format(v) for k, v in names.items()) # '{}'.format(i)
            col_name = '%s__%s_%s%s' % (name, out_name, set_name, '_%s'%(i) if len(input_names) > 1 else '')
            output[col_name] = (
                pd.Series(result[j] if isinstance(result, list) else result, index=prices.index) if is_pandas
                else result[j] if isinstance(result, list) else result
            )

    return _format_indicator_output(output, is_pandas)

########## tulipy ##########

def _tulip_indicator(name, prices, input_names=None, nan_offset=0, **kwargs):
    # optional import
    try:
        import tulipy
    except ImportError:
        logging.getLogger(__name__).debug('tulipy failed to import or does not exist, continuing')
        return None

    # setup
    prices = _name_prices(prices)
    base_name = _get_indi_name(name)
    is_pandas = isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series)
    trim_nan_side = 'f' if nan_offset > 0 else 'b' if nan_offset < 0 else 'fb'
    indi_function = getattr(tulipy, base_name.lower())
    
    # get parameters
    indi_parameters = {
        name: kwargs[name] for name in list(inspect.signature(indi_function).parameters) if name in kwargs
    }
        # there are no defaults for tulipy, so all params must be present
        # this list also includes price params, which are ignored as long as kwargs does not have them
    
    # get input fields
    indi_fields = indi_function.inputs

    # get inputs
    input_names = _get_input_names(input_names, indi_fields, prices)
    output = {}
    for i, names in enumerate(input_names):
        indi_inputs = {
            field: (
                _trim_nan(_get_price_field(prices, field, input_names=names).as_matrix().astype(np.float64), trim=trim_nan_side) if is_pandas
                else _trim_nan(_get_price_field(prices, field, input_names=names).astype(np.float64), trim=trim_nan_side)
            ) for field in indi_fields
        }

        # get result
        result = indi_function(**indi_inputs, **indi_parameters)

        # format output
        for j, out_name in enumerate(indi_function.outputs):
            set_name = ':'.join('{}'.format(v) for k, v in names.items()) # '{}'.format(i)
            col_name = '%s__%s_%s%s' % (name, out_name, set_name, '_%s'%(i) if len(input_names) > 1 else '')
            arr = _pad_array(result[j] if isinstance(result, tuple) else result, nan_offset, len(prices))
            
            ### DEBUG ###
            # if col_name in output:
            #     print('DEBUG: col_name already exists in output')
            #     import pdb; pdb.set_trace()
            
            output[col_name] = (
                pd.Series(arr, index=prices.index) if is_pandas
                else arr
            )

    return _format_indicator_output(output, is_pandas)

def _trim_nan(arr, trim='f'): # trim='fb'
    output = np.copy(arr)

    if 'f' in trim:
        trim_count = 0
        for val in arr:
            if np.isnan(val): trim_count += 1
            else: break
        output = output[trim_count:]

    if 'b' in trim:
        trim_count = 0
        reverse_arr = np.flipud(arr)
        for val in reverse_arr:
            if np.isnan(val): trim_count += 1
            else: break
        output = output[0:len(output)-trim_count]

    return output

def _pad_array(arr, pad_offset, expected_length):
    prepend, pad_offset = pad_offset >= 0, abs(pad_offset)

    insert_count = expected_length-len(arr)
    replace_count = pad_offset-insert_count

    if insert_count <= 0 and replace_count <= 0: return arr
    else: output = np.copy(arr)

    if prepend:
        if replace_count > 0: output[0:replace_count] = np.nan
        if insert_count > 0: output = np.insert(output, 0, np.full(insert_count, np.nan))
    else:
        if replace_count > 0: output[-replace_count:] = np.nan
        if insert_count > 0: output = np.append(output, np.full(insert_count, np.nan))

    return output

########## transformer ##########

class IndicatorTransformer(BaseEstimator, TransformerMixin):
    """Apply indicators to a set of price data using talib, tulipy, or
    a supplied callable.

    Parameters
    ----------
    copy_prices: boolean, default=False
        Return original prices as part of the output.

    default_lib: string, options=['talib','tulipy'], default='talib'
        Whether to search for indicator in talib first, or tulipy. If an
        indicator doesn't exist in the default library, then it's searched
        for in the second library. If an indicator pair is passed with
        `callable_`, the callable takes priority over both libraries.

    kwargs: string=dict pairs, e.g., `indicator name = {indicator params}`
        Arbitrary number of indicator pairs. The key string is the indicator
        name to search for in talib or tulipy, or an arbitrary name if `callable_`
        is passed. The name can be suffixed with `__x` in case of defining
        the indicator more than once; the suffix is ignored in processing.
        
        If value is True or {}, run the indicator with default parameters. 
        If value is False or None, skip that indicator.
        
        In the value dict, key is the setting to configure. If
        any settings are not specified in the value dict, they are run
        with their defaults.

        The following value dict keys have special meaning:

        * `callable_`: callable or string ['talib','tulipy'].
            Callable to process price data. Input parameters are 
            `(indi_name, prices, input_names={}, **indi_params)`.
            Output is array-like (n_samples,n_features). Other settings
            in the value dict are passed as arguments to this callable. If this value is 
            'talib' or 'tulipy', then force that library for the indicator.

        * `inputs_`: dict, (indicator price input name) = (replacement input name).
            Price column names to use for indicator inputs, if they should be different
            from the open/high/low/close/volume inputs as-is.

    Returns
    -------
    array-like (n_samples,n_features) dependent on the indicator(s) passed.
    Original prices are passed dependent on `copy_prices`.
    """
    def __init__(self, copy_prices=False, default_lib='talib', **kwargs):
        self.copy_prices = copy_prices
        self.default_lib = default_lib
        self.indicators = kwargs

    def __repr__(self):
        # needed to add indicators to the output
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint({**self.get_params(deep=False), **self.indicators},
                                               offset=len(class_name),),)

    def fit(self, x, y=None):
        return self

    def transform(self, prices):
        is_pandas = isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series)
        results = self._run_indicators(prices, self.indicators)

        if self.copy_prices:
            results['prices'] = prices

        if is_pandas:
            output = pd.concat([results[name] for name in results if results[name] is not None], axis=1)
        else:
            output = np.concatenate([results[name] for name in results if results[name] is not None], axis=1)

        return output

    def _run_indicators(self, prices, indicators):
        output = {}
        for indi_name, indi_params in indicators.items():
            # evaluate indi_params: False means skip, True means empty dict
            if indi_params is None or indi_params is False:
                continue
            elif indi_params is True or not isinstance(indi_params, dict):
                indi_params = {}

            # remove __x suffix for callable search
            base_indi_name = _get_indi_name(indi_name)

            # resolve callable
            # callable_ is a callable or forced talib/tulip
            # if callable_ does not exist, _get_indicator_callable searches talib and tulip
            # order depending on default_lib
            indi_src = self._get_reserved_param(indi_params, name_list=['callable_','_callable','lib_','_lib'])
            indi_callable = _get_indicator_callable(base_indi_name, src=indi_src, default_lib=self.default_lib)
            input_names = self._get_reserved_param(indi_params, name_list=['input_','_input','inputs_','_inputs'])

            # get result
            if callable(indi_callable):
                output[indi_name] = indi_callable(indi_name, prices, input_names=input_names, **indi_params)
            else:
                raise ValueError('Indicator %s not found'%(indi_name))

        return output

    def _get_reserved_param(self, indi_params, name_list=[]):
        """Note: indi_params is modified during this method.
        """
        indi_src = None
        for name in name_list:
            if name in indi_params:
                indi_src = indi_params.pop(name, None)
                break

        return indi_src

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
import collections

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _rename_pandas
sys.path.pop(0)
# end parent submodules

class DeltaTransformer(BaseEstimator, TransformerMixin):
    """Transformer for shifting input arrays.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, start, stop, step=1, operation='diff', fixed_start=False, remainder=True, percent_inf_val=np.nan, percent_midpoint=False, direction_negative_dir=0, direction_positive_dir=1):
        """Create a DeltaTransformer object.

        Parameters
        ----------
        shift: integer, list, or range
            Number of elements, or list of numbers, to shift by. Positive means shift to previous values, negative means shift to next values.
        """
        self.start = start
        self.stop = stop
        self.step = -step if self.stop < self.start else step
        self.operation = operation
        self.fixed_start = fixed_start
        self.remainder = remainder
        self.percent_inf_val = percent_inf_val
        self.percent_midpoint = percent_midpoint
        self.direction_negative_dir = direction_negative_dir
        self.direction_positive_dir = direction_positive_dir

    def fit(self, X, y=None, **fit_params):
        """Fit the DeltaTransformer transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        return self

    def transform(self, X):
        """Transform data by shifting features

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """
        is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        X_base = X.copy() if is_pandas else np.copy(X)
        transformed_list = []
        operation = self.operation
        pairs = self._gen_range(self.start, self.stop, self.step, fixed_start=self.fixed_start, remainder=self.remainder)

        for pair in pairs:
            base = self._shift_array(X_base, pair[0])
            subtrahend = self._shift_array(X_base, pair[1])
            if operation in ['diff','difference','delta']: 
                arr = self._get_diff(base, subtrahend)
            elif operation == 'percent':
                arr = self._get_percent(base, subtrahend, inf_val=self.percent_inf_val, midpoint=self.percent_midpoint)
            elif operation in ['direction','dir']:
                arr = self._get_direction(base, subtrahend, negative_dir=self.direction_negative_dir, positive_dir=self.direction_positive_dir)
            elif operation in ['cross','crossover']:
                arr = self._get_crossover(base, subtrahend)
            elif operation == 'slope':
                arr = self._get_slope(base, subtrahend, pair)
            else:
                continue

            if is_pandas:
                _rename_pandas(arr, '__'+self._get_pair_name(operation, pair[0], pair[1]), inplace=True)
            transformed_list.append(arr)

        if len(transformed_list) > 1:
            if is_pandas:
                X_transformed = pd.concat(transformed_list, axis=1)
            else:
                X_transformed = np.concatenate(transformed_list, axis=1)
        elif len(transformed_list) == 1:
            X_transformed = transformed_list[0]
        else:
            X_transformed = pd.Series(name='__delta') if is_pandas else np.array()

        return X_transformed
    
    def _gen_range(self, start, stop, step, fixed_start=False, remainder=True):
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

    def _get_pair_name(self, mode, start, stop):
        return '{}_{}:{}'.format(mode, start, stop)

    def _shift_array(self, arr, num, fill_value=np.nan):
        is_pandas = isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series)
        num = round(num)

        if is_pandas:
            result = arr.shift(num)
            if fill_value is not np.nan:
                if num > 0:
                    result.iloc[:min(num+1, len(result))] = fill_value
                elif num < 0:
                    result.iloc[max(len(result)+num, 0):] = fill_value
        else:
            result = shift(arr, shift=num, cval=fill_value, mode='constant')

        return result

    def _get_percent(self, start_val, end_val, inf_val=None, midpoint=False):
        if midpoint: output = (end_val-start_val)/((end_val+start_val)/2) # midpoint calc, https://sciencing.com/calculate-growth-rate-percent-change-4532706.html
        else: output = (end_val-start_val)/start_val
        #output[output.isnull()] = 0 # NAN when end_val and start_val are both 0, so no change
        if not inf_val is None:
            output[np.isnan(output)] = inf_val
        return output

    def _get_diff(self, start_val, end_val):
        return end_val-start_val

    def _get_direction(self, start_val, end_val, negative_dir=0, positive_dir=1):
        dir_series = self._get_diff(start_val, end_val)
        dir_series[dir_series <= 0] = negative_dir
        dir_series[dir_series > 0] = positive_dir
        return dir_series

    def _get_crossover(self, start_val, end_val, dummies=False):
        cur_series = self._get_diff(start_val, end_val)
        past_series = cur_series.shift(1)
        crossdn_series = np.sign(past_series) > np.sign(cur_series)
        crossup_series = np.sign(past_series) < np.sign(cur_series)

        if dummies:
            return crossup_series.astype(int), crossdn_series.astype(int)
        else:
            cross_series = pandas.Series(0, index=cur_series.index)
            cross_series[crossup_series] = 1; cross_series[crossdn_series] = -1
            return cross_series

    def _get_slope(self, start_val, end_val, shift_pair = (0,1)):
        output = (end_val-start_val)/(shift_pair[1]-shift_pair[0])*-1 # offset numbers are opposite of where the points are on x axis
        #output[output.isnull()] = 0 # NAN when shift_pair['end'] and shift_pair['start'] are equal, which means straight line so no slope
        return output

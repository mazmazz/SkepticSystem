import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
import collections

class DeltaTransformer(BaseEstimator, TransformerMixin):
    """Transformer for shifting input arrays.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, start, stop, step=1, operation='diff', fixed_start=False, remainder=True, percent_inf_val=np.nan, percent_midpoint=False):
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
        #X = check_array(X)
        X_base = np.copy(X)
        transformed_list = []
        operation = self.operation
        pairs = self._gen_range(self.start, self.stop, self.step, fixed_start=self.fixed_start, remainder=self.remainder)

        for pair in pairs:
            base = self._shift_array(X_base, pair[0])
            subtrahend = self._shift_array(X_base, pair[1])
            if operation in ['diff','difference','delta']: 
                transformed_list.append(self._get_diff(base, subtrahend))
            elif operation == 'percent':
                transformed_list.append(self._get_percent(base, subtrahend, inf_val=self.percent_inf_val, midpoint=self.percent_midpoint))
            elif operation in ['direction','dir']:
                transformed_list.append(self._get_direction(base, subtrahend))
            elif operation in ['cross','crossover']:
                transformed_list.append(self._get_crossover(base, subtrahend))
            elif operation == 'slope':
                transformed_list.append(self._get_slope(base, subtrahend, pair))

        X_transformed = np.concatenate(transformed_list, axis=1)

        return X_transformed
    
    def _gen_range(self, start, stop, step, fixed_start=False, remainder=True):
        # https://stackoverflow.com/questions/14048728/generate-list-of-range-tuples-with-given-boundaries-in-python
        current = next_current = start
        while next_current < stop:
            next_current = next_current + step
            if next_current < stop:
                yield (current, next_current)
            elif remainder: # elif remainder and stop-1 > next_current-step:
                yield (current, stop) # yield (current, stop-1)
            else:
                break
            if not fixed_start: current = next_current

    def _shift_array(self, arr, num, fill_value=np.nan):
        # preallocate empty array and assign slice by chrisaycock
        # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result = arr
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

    def _get_direction(self, start_val, end_val):
        dir_series = self._get_diff(start_val, end_val)
        dir_series[dir_series <= 0] = self.parameters['negative_class']
        dir_series[dir_series > 0] = self.parameters['positive_class']
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

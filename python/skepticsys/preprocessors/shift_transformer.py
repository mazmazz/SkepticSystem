import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
import collections

class ShiftTransformer(BaseEstimator, TransformerMixin):
    """Transformer for shifting input arrays.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, shift, fill_value=np.nan, keep_features=False):
        """Create a ShiftTransformer object.

        Parameters
        ----------
        shift: integer, list, or range
            Number of elements, or list of numbers, to shift by. Positive means shift to previous values, negative means shift to next values.
        """
        self.shift = shift if isinstance(shift, collections.Iterable) else [shift]
        self.fill_value = fill_value
        self.keep_features = keep_features

    def fit(self, X, y=None, **fit_params):
        """Fit the ShiftTransformer transformer.

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
        shift = self.shift
        fill_value = self.fill_value

        #X = check_array(X)
        X_base = np.copy(X)
        transformed_list = [X_base] if self.keep_features else []

        for shift_i in shift:
            transformed_list.append(self._shift_array(X_base, shift_i, fill_value=fill_value))
        
        X_transformed = np.concatenate(transformed_list, axis=1)

        return X_transformed

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
    
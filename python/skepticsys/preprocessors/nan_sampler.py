import numpy as np
import pandas as pd
from imblearn.base import SamplerMixin
import collections

class NanSampler(SamplerMixin):
    """Remove NaN rows from array

    Parameters
    ----------
    drop_inf: boolean, default=False
        Remove inf rows in addition to NaN rows.
    """
    def __init__(self, drop_inf=False):
        self.drop_inf = drop_inf

    def fit(self, X, y, *args):
        return self

    def sample(self, X, y, *args):
        return self._sample(X, y, *args)

    def _sample(self, X, y, *args):
        drop_inf = self.drop_inf

        def is_pandas(arr):
            return isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series)

        def dims(arr):
            return len(arr.shape)

        arrs = [X, y, *[arg for arg in args if isinstance(arg, collections.Iterable)]]
        arr_bases = []
        merged_mask = None

        # find mask for each array, then merge into merged_mask
        for arr in arrs:
            if not isinstance(arr, collections.Iterable):
                arr_bases.append(None)
                continue
            
            if drop_inf:
                if is_pandas(arr):
                    arr_base = arr.replace([np.inf, -np.inf], np.nan)
                else:
                    arr_base = np.copy(arr)
                    arr_base[arr_base == np.inf] = np.nan
                    arr_base[arr_base == -(np.inf)] = np.nan
            else:
                arr_base = arr.copy() if is_pandas(arr) else np.copy(arr)
            
            arr_bases.append(arr_base)

            arr_mask = (
                ~(arr_base.isnull().T.any() if dims(arr_base) > 1 else arr_base.isnull()) if is_pandas(arr_base)
                else ~(np.isnan(arr_base).any(axis=1) if dims(arr_base) > 1 else np.isnan(arr_base))
            )

            merged_mask = arr_mask if merged_mask is None else np.logical_and(merged_mask, arr_mask)

        # apply merged mask to each array
        return tuple([
            arr_base[merged_mask] if is_pandas(arr_base) or isinstance(arr_base, np.ndarray)
                else None
            for arr_base in arr_bases
        ])

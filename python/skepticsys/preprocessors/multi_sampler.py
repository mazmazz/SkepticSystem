import numpy as np
import pandas as pd
from imblearn.base import SamplerMixin
from imblearn.pipeline import make_pipeline
import collections
import copy
import random

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessors import IndicatorTransformer, CopyTransformer, DeltaTransformer, ShiftTransformer, NanSampler
from pipeline import make_union
from utils import is_pandas, copy_arr
sys.path.pop(0)
# end parent submodules

class MultiSampler(SamplerMixin):
    """Sampler for processing multiple indicators and transforms.

    Parameters
    ----------
    indi_params : dict
        Dictionary of indicator parameters.

    drop_nan : boolean, default True
        Drop NaN rows from transformed prices.

    drop_inf : boolean, default False
        Drop inf rows from transformed prices, ONLY if drop_nan is True.

    unique_column_names : boolean, default True
        Ensure column names are unique, if outputting Pandas.
    """

    def __init__(self, indi_params, drop_nan=True, drop_inf=False, unique_column_names=True):
        """Create a MultiTransformer object."""
        self.indi_pipeline = self._make_indi_pipeline(**indi_params)
        self.drop_nan = drop_nan
        self.drop_inf = drop_inf
        self.unique_column_names = unique_column_names

    def fit(self, X, y=None, **fit_params):
        """Fit the MultiTransformer transformer.

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

    def sample(self, X, y, *args):
        return self._sample(X, y, *args)

    def _sample(self, X, y, *args):
        """Transform data by shifting features

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            Copied features.
        """
        X_indi = self.indi_pipeline.transform(X)

        if self.drop_nan:
            X_nan, y_nan = NanSampler(drop_inf=self.drop_inf).sample(X_indi, y)
        else:
            X_nan, y_nan = X_indi, y

        if self.unique_column_names:
            X_named, y_named = self._name_unique_columns(X_nan), y_nan
        else:
            X_named, y_named = X_nan, y_nan

        if y is None:
            return X_named
        else:
            return X_named, y_named

    def _make_indi_pipeline(
        self, **indi_params_
    ):
        master_union = []
        indi_params = copy.deepcopy(indi_params_)
        for indi in indi_params:
            if not bool(indi_params[indi]):
                continue

            main_params = indi_params[indi].pop('_params', None)
        
            for subindi in indi_params[indi]:
                if not bool(indi_params[indi][subindi]):
                    continue
                else:
                    trans_pipe = []

                # do ma for the subindi pipeline
                ### HACK ### 
                # Params dict copy is needed to fix an error where IndicatorTransformer stores an empty dict
                # instead of the params
                if '_ma' in indi_params[indi][subindi] and bool(indi_params[indi][subindi]['_ma']):
                    ma_params = indi_params[indi][subindi]['_ma']
                    pre = ma_params.pop('_pre', None)
                    if pre:
                        trans_pipe.append(IndicatorTransformer(**{'ma__pre': {**ma_params}}))
                        trans_pipe.append(IndicatorTransformer(**{indi: {**main_params}}))
                    else:
                        trans_pipe.append(IndicatorTransformer(**{indi: {**main_params}}))
                        trans_pipe.append(IndicatorTransformer(**{'ma__post': {**ma_params}}))
                else:
                    trans_pipe.append(IndicatorTransformer(**{indi: {**main_params}}))

                # do delta child pipelines
                if '_delta' in indi_params[indi][subindi] and bool(indi_params[indi][subindi]['_delta']):
                    delta_params = indi_params[indi][subindi]['_delta']
                    base = delta_params.pop('_base', None)
                    for inst in delta_params:
                        if not bool(delta_params[inst]):
                            continue
                        ma_params = delta_params[inst].pop('_ma', None)
                        shift_params = delta_params[inst].pop('_shift', None)

                        # copy current pipe and apply transformer
                        inst_pipe = copy.deepcopy(trans_pipe)
                        inst_pipe.append(DeltaTransformer(**delta_params[inst]))

                        # if ma is specified, do that
                        if bool(ma_params):
                            ma_params.pop('_pre', None)
                            inst_pipe.append(IndicatorTransformer(**{'ma__delta':{**ma_params}}))

                        # if shift is specified, do that
                        if bool(shift_params):
                            inst_pipe.append(ShiftTransformer(**shift_params))

                        if len(inst_pipe) == 1:
                            master_union.append(inst_pipe[0])
                        elif len(inst_pipe) > 1:
                            master_union.append(make_pipeline(*inst_pipe))
                        # else, don't append anything, continue
                    # if _base exists and is false, don't construct the subindi pipeline (non-delta)
                    if base is not None and not base:
                        continue
                    # else, continue constructing the subindi pipeline

                # do shift for the subindi pipeline
                if '_shift' in indi_params[indi][subindi] and bool(indi_params[indi][subindi]['_shift']):
                    trans_pipe.append(ShiftTransformer(**indi_params[indi][subindi]['_shift'], keep_features=True))

                if len(trans_pipe) == 1:
                    master_union.append(trans_pipe[0])
                elif len(trans_pipe) > 1:
                    master_union.append(make_pipeline(*trans_pipe))
                # else, don't append anything, continue
        
        if len(master_union) == 1:
            master_pipeline = master_union[0]
        elif len(master_union) > 1:
            master_pipeline = make_union(*master_union)
        else:
            master_pipeline = None

        if not bool(master_pipeline):
            raise ValueError('MultiTransformer pipeline has no transformers.')

        return master_pipeline

    def _name_unique_columns(self, X):
        prices_indi = copy_arr(X)

        if is_pandas(prices_indi):
            dup_cols = prices_indi.columns.get_duplicates()
            if len(dup_cols) > 0:
                dups = prices_indi.columns[prices_indi.columns.isin(dup_cols)]
                dup_vals = prices_indi.loc[:,prices_indi.columns.isin(dup_cols)]
                unq_vals = prices_indi.loc[:,~prices_indi.columns.isin(dup_cols)]
                fixed_dups = dups.map(lambda x: x+'__'+str(random.uniform(0,1)))
                dup_vals.columns = fixed_dups
                prices_indi = pd.concat([unq_vals, dup_vals], axis=1)

        return prices_indi

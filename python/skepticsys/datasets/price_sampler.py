import numpy as np
import pandas as pd
from imblearn.base import SamplerMixin
from load_prices import load_prices, get_target

class PriceSampler(SamplerMixin):
    def __init__(self
        , instrument
        , granularity
        , target_end_offset

        , start_date=None
        , end_date=None 
        , source='csv'

        , target_start_offset=-1
        , target_column='open'
        , target_operation='direction'
        
        , append=False
        , **kwargs
    ):
        self.instrument = instrument
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.prices_kwargs = kwargs

        self.end_offset = target_end_offset
        self.start_offset = target_start_offset
        self.column = target_column
        self.operation = target_operation

        self.append = append

    def fit(self, X=None, y=None):
        return self

    def sample(self, X=None, y=None):
        return self._sample(X, y)

    def _sample(self, X=None, y=None):
        X_is_pandas = not isinstance(X, np.ndarray)
        y_is_pandas = not isinstance(y, np.ndarray)
        prices = load_prices(self.instrument, self.granularity, start_date=self.start_date, end_date=self.end_date, source=self.source, **self.prices_kwargs)
        target = get_target(prices, self.end_offset, start_offset=self.start_offset, column=self.column, operation=self.operation)

        if not X_is_pandas: prices = prices.as_matrix()
        if not y_is_pandas: target = target.as_matrix()

        if self.append:
            ### TODO ### how to merge arrays with same index? Pandas combine_first()? can't do with numpy, also.
            prices = pd.concat([X, prices], axis=0) if X_is_pandas else np.concatenate([X, prices], axis=0)
            target = pd.concat([y, target], axis=0) if y_is_pandas else np.concatenate([y, target], axis=0)

        return (prices, target)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
import collections

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _get_price_field
sys.path.pop(0)
# end parent submodules

class ConvolutionTransformer(BaseEstimator, TransformerMixin):
    """Transformer for passing input data as a convolution matrix.

    Parameters
    ----------
    noise_ma_smoother: int, default 3
        MA period to smooth raw price data. <1 means do not smooth.

    longer_ma_smoother: int, default 6
        MA period to overlay on price data

    time_range: int, default 20
        X size of convolution matrix; number of bars to lookback

    price_range: int, default 20
        Y size of convolution matrix; total number of Y data points for all indicators
    """

    def __init__(noise_ma_smoother=3, longer_ma_smoother=6, time_range=20, price_range=20):
        self.noise_ma_smoother = noise_ma_smoother
        self.longer_ma_smoother = longer_ma_smoother
        self.time_range = time_range
        self.price_range = price_range

    def fit(self, X, y=None, **fit_params):
        """Fit the ConvolutionTransformer transformer.

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
        """Transform data into a convolution matrix

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            Copied features.
        """
        # if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        #     return X.copy()
        # else:
        #     return np.copy(X)
        return self._do_transform(X)

    def _do_transform(stock_data):
        noise_ma_smoother = self.noise_ma_smoother # 3
        longer_ma_smoother = self.longer_ma_smoother # 6
        TIME_RANGE = self.time_range # 20
        PRICE_RANGE = self.price_range # 20

        # split image horizontally into two sections - top and bottom sections
        half_scale_size = int(PRICE_RANGE/2)

        x_train = None
        y_train = []
        train_data_xgboost = []

        # add Moving Averages to all lists and back fill resulting first NAs to last known value
        # BUT: if you have enough data, do not apply MA
        stock_closes = pd.rolling_mean(_get_price_field(stock_data, 'close'), window = noise_ma_smoother) 
        stock_closes = stock_closes.fillna(method='bfill')  
        stock_closes =  list(stock_closes.values)
        stock_opens = pd.rolling_mean(_get_price_field(stock_data, 'open'), window = noise_ma_smoother)
        stock_opens = stock_opens.fillna(method='bfill')  
        stock_opens =  list(stock_opens.values)
    
        close_minus_open = list(np.array(stock_closes) - np.array(stock_opens))

        # lets add a rolling average as an overlay indicator - back fill the missing
        # first five values with the first available avg price
        stock_closes_rolling_avg = pd.rolling_mean(_get_price_field(stock_data, 'close'), window = longer_ma_smoother)
        stock_closes_rolling_avg = stock_closes_rolling_avg.fillna(method='bfill')  
        stock_closes_rolling_avg =  list(stock_closes_rolling_avg.values)

        if noise_ma_smoother + 1 > len(stock_closes):
            raise ValueError('Not enough prices {} vs noise_ma_smoother {}'.format(len(stock_closes), noise_ma_smoother + 1))

        for cnt in range(noise_ma_smoother + 1, len(stock_closes)):
            if (cnt < TIME_RANGE):
                continue
            
            # start making images
            graph_open = list(np.round(scale_list(stock_opens[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
            graph_close_minus_open = list(np.round(scale_list(close_minus_open[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
            
            # scale both close and close MA together
            close_data_together = list(np.round(scale_list(list(stock_closes[cnt-TIME_RANGE:cnt]) + 
                list(stock_closes_rolling_avg[cnt-TIME_RANGE:cnt]), 0, half_scale_size-1),0))
            graph_close = close_data_together[0:PRICE_RANGE]
            graph_close_ma = close_data_together[PRICE_RANGE:] 

            # outcome: if this specific stock close > MA close, it's a 1 outcome
            outcome = None
            if (cnt < len(stock_closes) - 2): # -1):
                outcome = 0
                if stock_closes[cnt+2] > stock_closes[cnt+1] and stock_closes[cnt+1] > stock_closes_rolling_avg[cnt+1]:
                    outcome = 1

            blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE))
            x_ind = 0
            for ma, c in zip(graph_close_ma, graph_close):
                blank_matrix_close[int(ma), x_ind] = 1 
                blank_matrix_close[int(c), x_ind] = 2  
                x_ind += 1

            # flip x scale dollars so high number is atop, low number at bottom - cosmetic, humans only
            blank_matrix_close = blank_matrix_close[::-1]

            # store image data into matrix DATA_SIZE*DATA_SIZE
            blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE))
            x_ind = 0
            for v in graph_close_minus_open:
                blank_matrix_diff[int(v), x_ind] = 3  
                x_ind += 1
            # flip x scale so high number is atop, low number at bottom - cosmetic, humans only
            blank_matrix_diff = blank_matrix_diff[::-1]

            blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff]) 

            if 1==1:
                # graphed on matrix
                plt.imshow(blank_matrix)
                plt.show()

                # straight timeseries 
                plt.plot(graph_close, color='black')
                plt.show()

            if x_train is None:
                x_train = [blank_matrix]
            else:
                x_train = np.vstack([x_train, [blank_matrix]])
            y_train.append(outcome)

            train_data_xgboost.append(graph_close_ma + graph_close + graph_close_minus_open + [outcome])
                    
        pass
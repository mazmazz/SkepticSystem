from sklearn.metrics.scorer import _BaseScorer # make_scorer #, _ProbaScorer, _PredictScorer
from sklearn.base import BaseEstimator
from utils import arr_to_datetime
import backtrader as bt
import pandas as pd
import numpy as np
import sys

class BacktraderScorer(_BaseScorer):
    """Scorer using a Backtrader Cerebro instance.

    Parameters
    ----------
    cerebro: Cerebro instance
        Cerebro instance, prepared with necessary brokers, data feeds, and analyzers,
        but no strategies.

        Strategies are added and replaced on every scoring call, using y_pred as input
        for the scoring.

    strategy_class: class or list of classes
        Name of strategy class, or a list for multiple strategies. Strategy
        must have an input that accepts y_pred: a Pandas Series with datetime
        index.

    strategy_pred_kw: string or list of strings
        Name of kwarg to specify y_pred, or a list for multiple strategies.

    strategy_kwargs: dict or list of dicts, optional
        Dictionary of kwargs to pass to strategy, or a list for multiple strategies.

    analyzer_name: string, optional
        Name of analyzer to read from. If None, read the account P&L at end of test.
        Analyzer must return a single value in get_analysis(), or specify `analysis_key`
        to return a specific key from get_analysis(). Else, ValueError is raised.

    analysis_key: string or list of strings, optional
        Dictionary key to read from analyzer's get_analysis() output. Either a single
        string or a list of strings to go down a dict hierarchy. If empty list,
        return the output dict as-is. If None, analyzer must return only one metric.

    analyzer_callback: callable, optional
        Callable to process analysis metric, useful if multiple metrics are returned from,
        e.g., multiple strategies. If None, metric is returned as-is.

    initial_cash: float, default=100000.
        Initial cash to set for broker on each run.

    process_pred: bool, default=True
        Convert y_pred to Pandas Series with DatetimeIndex. Else, pass y_pred to Cerebro as-is.

    Returns
    -------
    Backtrader score of either account P&L at end of test, or a metric determined from analyzer_name and analyzer_callback.
    """
    def __init__(self, cerebro, strategy_class, strategy_pred_kw, strategy_kwargs=None, analyzer_name=None, analysis_key=None, analyzer_callback=None, initial_cash=100000., process_pred=True):
        self.cerebro = cerebro
        self.strategy_class = strategy_class
        self.strategy_pred_kw = strategy_pred_kw
        self.strategy_kwargs = strategy_kwargs
        self.analyzer_name = analyzer_name
        self.analysis_key = analysis_key
        self.analyzer_callback = analyzer_callback
        self.initial_cash = initial_cash
        self.process_pred = process_pred

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict_proba(X)
        return self._bt_score(y_pred, y_true=y_true, process_pred=self.process_pred)

    def _bt_score(self, y_pred, y_true=None):
        """Score y_pred with Backtrader result.

        Parameters
        ----------
        y_pred: array-like, shape (n_samples,)
            Array to score
        
        y_true: array-like, shape (n_samples,), optional
            Prediction truth. Used to convert y_pred to Pandas Series with this
            array's index, if process_pred=True

        Returns
        -------
        Backtrader account P&L, or an analysis metric if specified on init.
        """
        # setup
        y_pred = arr_to_datetime(y_pred, y_true=y_true) if self.process_pred else y_pred
        cerebro = self._reset_cerebro(y_pred)
        analyzer_name = self.analyzer_name
        analysis_key = self.analysis_key
        starting_balance = cerebro.broker.getvalue()
        
        strats = cerebro.run()

        if analyzer_name is None:
            # return ending P&L by default
            metric = cerebro.broker.getvalue() - starting_balance
        else:
            if len(strats) > 1: 
                metric = []
            for strat in strats:
                analysis = getattr(strat.analyzers, analyzer_name).get_analysis()
                if len(strats) == 1:
                    metric = self._get_analysis_metric(analysis, analysis_key)
                else:
                    metric.append(self._get_analysis_metric(analysis, analysis_key))

        if callable(self.analyzer_callback):
            return self.analyzer_callback(metric)
        else:
            return metric

    def _reset_cerebro(self, y_pred):
        # set initial cash
        self.cerebro.broker.set_cash(self.initial_cash)

        # reset strats
        self.cerebro.strats = []

        # add new strats
        strategy_class = self.strategy_class if isinstance(self.strategy_class, list) else [self.strategy_class]
        strategy_pred_kw = self.strategy_pred_kw if isinstance(self.strategy_pred_kw, list) else [self.strategy_pred_kw]
        strategy_kwargs = self.strategy_kwargs if isinstance(self.strategy_kwargs, list) else [self.strategy_kwargs]

        if len(strategy_class) != len(strategy_pred_kw) or len(strategy_pred_kw) != len(strategy_kwargs) or len(strategy_kwargs) != len(strategy_class):
            raise ValueError('Strategy parameters must be same list length. strategy_class: %s, strategy_pred_kw: %s, strategy_kwargs: %s' % (len(strategy_class), len(strategy_pred_kw), len(strategy_kwargs)))

        for i, strat_class in enumerate(strategy_class):
            strat_kwargs = strategy_kwargs[i] if isinstance(strategy_kwargs[i], dict) else {}
            strat_kwargs[strategy_pred_kw[i]] = y_pred
            self.cerebro.addstrategy(strat_class, **strategy_kwargs[i])
        
        return self.cerebro

    def _get_analysis_metric(self, analysis, metric_key):
        if not isinstance(metric_key, list):
            metric_key = [metric_key]

        analysis_obj = analysis
        for key in metric_key:
            if key in analysis_obj:
                analysis_obj = analysis_obj[key]
            elif len(metric_key) == 1 and key is None:
                # special case where metric_key is None returns single value from analysis
                analysis_obj = next(iter(analysis.values()))
            elif len(metric_key) > 1 and key is None:
                raise ValueError('Analyzer returned %s metrics; must return only 1, or specify valid analysis_key.' % (len(analysis)))
            else:
                raise ValueError('Key %s does not exist in analyzer' % (key))
        
        return analysis_obj

def backtrader_score(y_true, y_pred
    , cerebro=None
    , strategy_class=None
    , strategy_pred_kw=None
    , strategy_kwargs=None
    , analyzer_name=None
    , analysis_key=None
    , analyzer_callback=None
    , initial_cash=100000.
    , process_pred=True
):
    """Metric callable for ProfitScorer. Recommended to use `ProfitScorer` directly."""
    scorer = BacktraderScorer(
        cerebro
        , strategy_class=strategy_class
        , strategy_pred_kw=strategy_pred_kw
        , strategy_kwargs=strategy_kwargs
        , analyzer_name=analyzer_name
        , analysis_key=analysis_key
        , analyzer_callback=analyzer_callback
        , initial_cash=initial_cash
        , process_pred=process_pred
    )

    return scorer._bt_score(y_pred, y_true=y_true)

import numpy as np
import pandas as pd
import hyperopt as hp
from imblearn.pipeline import make_pipeline
from collections import OrderedDict
import copy
from xgboost import XGBClassifier
import random
import sklearn.metrics as skm
from sklearn.utils.sparsefuncs import count_nonzero
import backtrader as bt
import uuid
import datetime
import traceback

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_validation import SingleSplit, WindowSplit
from calibration import ClassifierCV
from metrics import BacktraderScorer
from trading import SeriesStrategy, BasicTradeStats
from preprocessors import IndicatorTransformer, CopyTransformer, DeltaTransformer, ShiftTransformer, NanSampler
from datasets import load_prices, get_target
from pipeline import make_union
from utils import arr_to_datetime
sys.path.pop(0)
# end parent submodules

def do_candidate(params):
    try:
        return do_fit_predict(params)
    except Exception as e:
        # https://stackoverflow.com/a/1278740
        #raise e
        traceback.print_exc()
        fname = os.path.split(sys.exc_info()[-1].tb_frame.f_code.co_filename)[1]
        msg = '%s, %s, %s | %s' % (sys.exc_info()[0].__name__, fname, sys.exc_info()[-1].tb_lineno, str(e))
        out = fail_trial('Exception: %s'%(msg))
        #print(out)
        return out

def fail_trial(msg, **data):
    out = {'status': hp.STATUS_FAIL, 'id': str(uuid.uuid4()), 'date': str(datetime.datetime.now()), 'msg': msg}
    for k in data:
        out[k] = data[k]
    print('Trial failed: {}'.format(msg))
    return out

def do_fit_predict(params):
    #setup
    nans = NanSampler(drop_inf=False)
    cv = get_cv(**params['cv__params'])
    print('='*48)

    # load prices
    prices, target = do_data(**params['data__params'])
    prices_trade = prices.copy()
    target_trade = target.copy()

    # get cv model and check validity
    prices_model, target_model = nans.sample(prices, target)

    cv_model = list(cv.split(prices_model))
    for i, (train, test) in enumerate(cv_model):
        if len(train) == 0 or len(test) == 0:
            return fail_trial('CV invalid: set size is 0', train_len=len(train), test_len=len(test))
        else:
            print('Model CV {} freqs: {}'.format(i, target_model.iloc[test].value_counts().to_dict()))

    # do indicators
    print('Doing indicators') ##############
    indi_pipeline = do_indicators(**params['indicator__params'])
    if not bool(indi_pipeline):
        return fail_trial('Indicator pipeline: No transformers')
    prices_indi = indi_pipeline.transform(prices)

    # drop nan
    prices, target = nans.sample(prices_indi, target)

    if len(prices) == 0:
        return fail_trial('Nan pipeline: No prices exist after transformation', shape=prices.shape)

    # make all column names unique
    dup_cols = prices.columns.get_duplicates()
    if len(dup_cols) > 0:
        dups = prices.columns[prices.columns.isin(dup_cols)]
        dup_vals = prices.loc[:,prices.columns.isin(dup_cols)]
        unq_vals = prices.loc[:,~prices.columns.isin(dup_cols)]
        fixed_dups = dups.map(lambda x: x+'__'+str(random.uniform(0,1)))
        dup_vals.columns = fixed_dups
        prices = pd.concat([unq_vals, dup_vals], axis=1)

    # do classifier
    print('Doing classifier') ##############
    print('Prices shape: {}'.format(prices.shape if hasattr(prices, 'shape') else None))

    clf = do_classifier(**params['classifier__params'])

    # split CV
    cv_split = list(cv.split(prices))
    ### TODO ### More sophisticated CV model checking
    if len(cv_split) != len(cv_model):
        return fail_trial('CV invalid: does not match model split count', split_len=len(cv_split), model_len=len(cv_model))

    fail_reason = {}
    def check_split_model(X_train, y_train, X_test, y_test, i):
        # validate CV
        train_model, test_model = cv_model[i][0], cv_model[i][1]

        print('CV {} size: {}, {}'.format(i, len(X_train), len(X_test))) ##############
        if len(X_train) == 0 or len(X_test) == 0:
            for k, v in fail_trial('CV invalid: set size is 0', train_len=len(X_train), test_len=len(X_test)).items():
                fail_reason[k] = v
            return False

        if len(X_test) != len(test_model):
            for k, v in fail_trial('CV invalid: test len does not match model', test_len=len(X_test), model_len=len(test_model)).items():
                fail_reason[k] = v
            return False

        # count CV frequencies
        y_model = target_trade.iloc[test_model]
        test_counts, model_counts = {k: v for k, v in zip(*[x.tolist() for x in np.unique(y_test, return_counts=True)])}, {k: v for k, v in zip(*[x.tolist() for x in np.unique(y_model, return_counts=True)])}
        print('CV {} freqs: {} | Model freqs: {}'.format(i, test_counts, model_counts))
        if test_counts != model_counts:
            for k, v in fail_trial('CV invalid: test freqs do not match model', test_freqs=test_counts, model_freqs=model_counts).items():
                fail_reason[k] = v
            return False

        return True

    clf_cv = ClassifierCV(clf, cv=cv_split, prefit_callback=check_split_model) #, prefit_params={'train_model': cv_model[0], 'test_model': cv_model[1]})
    
    try:
        clf_cv.fit(prices, target)
    except Exception as e:
        traceback.print_exc()
        if len(fail_reason) > 0:
            return fail_reason
        else:
            return fail_trial('ClassifierCV error: %s'%(str(e)))

    # score
    print('Scoring') ##############
    agg_method = 'concatenate'

    acc = clf_cv.score_cv(skm.accuracy_score, aggregate=agg_method)
    precision, recall, fscore, support = clf_cv.score_cv(skm.precision_recall_fscore_support, aggregate=agg_method)
    brier = clf_cv.score_cv(skm.brier_score_loss, aggregate=agg_method, proba_positive=True)
    logloss = clf_cv.score_cv(skm.log_loss, aggregate=agg_method)

    # prep backtrader score
    end_offset = params['data__params']['end_target']

    y_test = clf_cv.y_true
    y_pred = clf_cv.y_pred

    try:
        start_loc = prices_trade.index.get_loc(y_test.index[0])
    except KeyError:
        start_loc = 0
    try:
        end_loc = min(prices_trade.index.get_loc(y_test.index[-1])-end_offset, len(prices_trade)-1)
    except KeyError:
        end_loc = len(prices_trade)-1

    y_prices = prices_trade.iloc[start_loc:end_loc+1,:]

    pnl, trade_stats = do_backtest(y_pred, y_test, y_prices)

    # compile scores
    loss = logloss #-acc # brier # -pnl

    out = {
        'status': hp.STATUS_OK
        , 'loss': loss
        , 'id': str(uuid.uuid4())
        , 'date': str(datetime.datetime.now())
        , 'trade_stats': trade_stats
        , 'pnl': pnl
        , 'brier': brier
        , 'logloss': logloss
        , 'accuracy': acc
        , 'precision': list(precision if precision is not None else [])
        , 'recall': list(recall if recall is not None else [])
        #, 'fscore': list(fscore if fscore is not None else [])
        , 'support': list(support if support is not None else [])
        , 'shape': prices.shape if hasattr(prices,'shape') else 'No shape? ' + str(type(prices))
    }

    import pprint; pprint.pprint(out)
    return out

def do_data(
    instrument
    , granularity
    , end_target
    , source='csv'
    , start_index=None
    , end_index=None
    , sample_len=None
    , dir='.'
):
    prices = load_prices(instrument, granularity, start_index=start_index, end_index=end_index, source=source, sample_len=sample_len, dir=dir)
    target = get_target(prices, end_target)
    return prices, target

def do_indicators(
    **indi_params
):
    master_union = []
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
        return master_union[0]
    elif len(master_union) > 1:
        return make_union(*master_union)
    else:
        return None

def do_classifier(
    **classifier_params
):
    master_pieces = {}
    for _, pipe_params in classifier_params.items():
        if not bool(pipe_params):
            continue

        order_base = pipe_params.pop('_order_base', 0.)
        order_factor = pipe_params.pop('_order_factor', 0.)
        order_key = float(order_base)*float(order_factor)
        while order_key in master_pieces:
            order_key += random.uniform(-1,1)

        pipe_pieces = []
        for clf_name, clf_params in pipe_params.items():
            if not bool(clf_params):
                continue

            if clf_name == 'xgb':
                pipe_pieces.append(XGBClassifier(**clf_params))

        if len(pipe_pieces) == 1:
            master_pieces[order_key] = pipe_pieces[0]
        elif len(pipe_pieces) > 1:
            master_pieces[order_key] = make_pipeline(*pipe_pieces)

    if len(master_pieces) == 1:
        return list(master_pieces.items())[0][-1] # value
    elif len(master_pieces) > 1:
        return make_pipeline(*[master_pieces[k] for k in sorted(list(master_pieces))])
    else:
        return None

def do_backtest(
    y_pred, y_true, y_prices
    , initial_balance = 100000.
):
    y_prices = arr_to_datetime(y_prices, y_true=y_true)
    
    # make cerebro for BacktraderScorer
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_balance)
    data = bt.feeds.PandasData(dataname=y_prices, openinterest=None)
    data2 = bt.feeds.PandasData(dataname=y_prices, openinterest=None)
    cerebro.adddata(data, name='LongFeed')
    cerebro.adddata(data2, name='ShortFeed')
    cerebro.addanalyzer(BasicTradeStats, useStandardPrint=True, useStandardDict=True, _name='BasicStats')

    # make scorers
    bts = BacktraderScorer(cerebro
        , SeriesStrategy, 'signals', strategy_kwargs={'tradeintervalbars':0, 'tradeexpirebars':60, 'stake':1}
        , analyzer_name='BasicStats', analysis_key=[] #['all','stats','kellyPercent']
        , initial_cash=initial_balance
    )

    # get score
    results = bts._bt_score(y_pred, y_true=y_true)
    results['won'].pop('streak')
    results['lost'].pop('streak')
    pnl = results['all']['pnl']['total']

    return pnl, results

def get_cv(**cv_params):
    if 'cv' in cv_params:
        return cv_params['cv'](**cv_params['params'])
    elif 'single_split' in cv_params:
        return SingleSplit(test_size=cv_params['single_split'])
    else:
        return None
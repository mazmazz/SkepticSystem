import numpy as np
import pandas as pd
import hyperopt as hp
from hyperopt import fmin, space_eval, tpe, Trials
from imblearn.pipeline import make_pipeline
from collections import OrderedDict, Iterable
import copy
from xgboost import XGBClassifier
import random
import sklearn.metrics as skm
from sklearn.utils.sparsefuncs import count_nonzero
import backtrader as bt
import backtrader.analyzers as bta
import uuid
import datetime
import traceback
import pprint
from sklearn.base import clone
import math
import statistics
from space import get_space

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_validation import SingleSplit, WindowSplit
from estimators import ClassifierCV, CalibratedClassifierCV, ThresholdClassifierCV, CutoffClassifierCV
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
        if params['cv__params']['doing_transforms'] and params['cv__params']['transforms'] is None:
            return fail_trial('Transform phase must specify a transform step')

        super_threshold_level = params['meta__params']['super_threshold']
        super_threshold_field = params['meta__params']['super_field']

        if False and not params['cv__params']['doing_transforms']: # testing
            base_result = {'base':{super_threshold_field:super_threshold_level}, 'status': hp.STATUS_OK, 'loss': random.randint(0,99)}
        else:
            base_result = do_fit_predict(params, super_threshold_level=super_threshold_level, super_threshold_field=super_threshold_field)

        if 'base' in base_result and super_threshold_field in base_result['base']:
            # todo:
            # Smarter super_field (#20)
            # Secondary super_threshold (parameterize check for verify score)
            if 'verify_score' in base_result and base_result['verify_score'] > 0.6:
                base_result['super_score'] = base_result['base'][super_threshold_field]
            else:
                base_result['super_score'] = None

        if (not params['cv__params']['doing_transforms'] 
            and isinstance(base_result, dict) 
            and base_result['status'] == hp.STATUS_OK
            and 'base' in base_result
            and super_threshold_field in base_result['base']
            and base_result['base'][super_threshold_field] >= super_threshold_level
        ):
            try:
                transform_result = do_transform_optimization(params, limit=params['meta__params']['transform_limit'], super_threshold=base_result['base'][super_threshold_field]-0.05, super_field=super_threshold_field)
                    # super at least within 0.05 of original score
                base_result['transform'] = transform_result
                base_result['transform_score'] = transform_result['super_score'] if isinstance(transform_result, dict) and 'super_score' in transform_result else None

                if isinstance(transform_result, dict) and 'super_score' in transform_result and (base_result['super_score'] is None or transform_result['super_score'] > base_result['super_score']):
                    if 'verify_score' in transform_result and transform_result['verify_score'] > 0.6:
                        base_result['super_score'] = transform_result['super_score'] # todo: transform_score and super_score separate?
            except Exception as e:
                traceback.print_exc()
                print('Error optimizing transform: {}'.format(e))
                print('Passing original candidate as-is')

        return base_result
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

def do_fit_predict(params, super_threshold_level=0.65, super_threshold_field='accuracy'):
    #setup
    nans = NanSampler(drop_inf=False)
    print('='*48)

    # load prices
    prices, target = do_data(params['data__params'], params['cv__params'])
    prices_trade = prices.copy()
    target_trade = target.copy()

    # get cv model and check validity
    prices_model, target_model = nans.sample(prices, target)

    # get cv
    cv = get_cv(prices_model, params['data__params'], params['cv__params'])
    cv_verify = get_cv(prices_model, params['data__params'], params['cv__params'], do_verify=True)

    ##### start cv logging #####
    print('CV parameters')
    pprint.pprint(params['cv__params'], indent=4)
    for i, cv_unit in enumerate(cv):
        print('CV {}: {}'.format(i, str(cv_unit)))
    for i, cv_unit in enumerate(cv_verify):
        for j, cv_subunit in enumerate(cv_unit):
            print('Verify CV {}-{}: {}'.format(i, j, str(cv_subunit)))

    print('Sample_len query parameters')
    pprint.pprint(get_sample_len(params['data__params'], params['cv__params']))

    print('Price size: {}{}'.format(len(prices_model)
        , ' | From {} to {}'.format(prices_model.index[0], prices_model.index[-1]) if isinstance(prices_model, pd.DataFrame) else ''
    ))
    ##### end logging #####

    ### todo: CV model

    # do indicators
    print('Doing indicators') ##############
    indi_pipeline = do_indicators(**params['indicator__params'])
    if not bool(indi_pipeline):
        return fail_trial('Indicator pipeline: No transformers')
    prices_indi = indi_pipeline.transform(prices_model)
    # drop nan
    prices_indi, target_indi = nans.sample(prices_indi, target_model)

    if len(prices_indi) == 0:
        return fail_trial('Nan pipeline: No prices exist after transformation', shape=prices_indi.shape)

    # make all column names unique
    dup_cols = prices_indi.columns.get_duplicates()
    if len(dup_cols) > 0:
        dups = prices_indi.columns[prices_indi.columns.isin(dup_cols)]
        dup_vals = prices_indi.loc[:,prices_indi.columns.isin(dup_cols)]
        unq_vals = prices_indi.loc[:,~prices_indi.columns.isin(dup_cols)]
        fixed_dups = dups.map(lambda x: x+'__'+str(random.uniform(0,1)))
        dup_vals.columns = fixed_dups
        prices_indi = pd.concat([unq_vals, dup_vals], axis=1)

    # do classifier
    print('Doing classifier') ##############
    print('Prices shape: {}{}'.format(prices_indi.shape if hasattr(prices_indi, 'shape') else None
        , ' | From {} to {}'.format(prices_indi.index[0], prices_indi.index[-1]) if isinstance(prices_indi, pd.DataFrame) else ''
    ))

    ### todo: split CV

    clf = do_classifier(params['cv__params'], **params['classifier__params'])

    score_base = do_cv_fit_score(prices_model, target_model, prices_indi, target_indi, prices_trade, target_trade
                                 , cv, params, clf_method=do_classifier_transforms, score_backtest=False
                                 , base_clf=clf, cv_list=cv, cv_params=params['cv__params'], base_only=False
                                 )

    if score_base['status'] == hp.STATUS_FAIL:
        return score_base
    elif score_base[super_threshold_field] >= super_threshold_level:
        print('Base %s: %s\nDoing verification...' % (super_threshold_field, score_base[super_threshold_field]))

        out = {
            'status': score_base['status']
            , 'loss': score_base['loss']
            , 'base': score_base
            , 'transform': None
            , 'verify': {}
        }

        # verify
        for i, cv_verify_unit in enumerate(cv_verify):
            assert len(cv_verify_unit) <= 2

            if len(cv_verify_unit) > 1: # has a post unit
                subfactors = [params['cv__params']['verify_factor'][i], 1-params['cv__params']['verify_factor'][i]]
            else:
                subfactors = [params['cv__params']['verify_factor'][i]]

            print('Subfactors {}'.format(subfactors))
            factor = params['cv__params']['verify_factor'][i]
            out['verify'][factor] = {}

            for j, cv_subverify in enumerate(cv_verify_unit):
                print('Doing subfactor {}'.format(subfactors[j]))
                is_pre = j == 0
                clf = do_classifier(params['cv__params'], **params['classifier__params'])
                clf_verify = do_cv_fit(prices_model, target_model, prices_indi, target_indi, prices_trade, target_trade
                                    , cv_subverify, params, clf_method=do_classifier_transforms
                                    , base_clf=clf, cv_list=cv_subverify, cv_params=params['cv__params'], base_only=False
                                    )
                if isinstance(clf_verify, dict) and clf_verify['status'] == hp.STATUS_FAIL:
                    return clf_verify
                score_verify = do_score(clf_verify, params, prices_indi, prices_trade, backtest=not is_pre)
                score_verify['verify_post'] = not is_pre
                score_verify['verify_factor'] = subfactors[j]
                out['verify'][factor]['pre' if is_pre else 'post'] = score_verify
                print('Subfactor %s Accuracy: %s'%(subfactors[j], score_verify['accuracy']))

        out['verify_best_factor'] = get_best_verify_key(out['verify'])
        out['verify_best_accuracy'] = get_best_verify_accuracy(out['verify'])
        out['verify_avg_accuracy'] = get_avg_verify_accuracy(out['verify'])
        out['verify_score'] = out['verify_avg_accuracy']
        print('Best verify factor: %s'%(out['verify_best_factor']))
        print('Best verify accuracy: %s'%(out['verify_best_accuracy']))
        print('Average verify accuracy: %s'%(out['verify_avg_accuracy']))
    else:
        print('Base %s: %s\nFinishing...' % (super_threshold_field, score_base[super_threshold_field]))
        out = {
            'status': score_base['status']
            , 'loss': score_base['loss']
            , 'base': score_base
            , 'transform': None
            , 'verify': None
        }
    
    # add metadata
    out['meta'] = {
        'data': params['data__params']
        , 'cv': params['cv__params']
    }

    # score
    pprint.pprint(out)
    return out

def do_cv_fit(prices_model, target_model, prices_indi, target_indi, prices_trade, target_trade, cv, params, clf_method, **clf_params):
    cv_model = list(cv[-1].split(prices_model)) # concerned only with the lastmost CV
    for i, (train, test) in enumerate(cv_model):
        if len(train) == 0 or len(test) == 0:
            return fail_trial('CV invalid: set size is 0', train_len=len(train), test_len=len(test))
        else:
            print('Model CV Split {} freqs: {}{}'.format(i, target_model.iloc[test].value_counts().to_dict()
                , ' | From {} to {}'.format(target_model.iloc[test].index[0], target_model.iloc[test].index[-1]) if isinstance(target_model, pd.Series) else ''
            ))
    
    # split CV
    cv_split = list(cv[-1].split(prices_indi)) # concerned only with the lastmost CV
    ### TODO ### More sophisticated CV model checking
    if len(cv_split) != len(cv_model):
        return fail_trial('CV invalid: %s does not match model split count %s' % (len(cv_split), len(cv_model)), split_len=len(cv_split), model_len=len(cv_model))

    fail_reason = {}
    def check_split_model(X_train, y_train, X_test, y_test, i, cv_model_test):
        # validate CV
        train_model, test_model = cv_model_test[i][0], cv_model_test[i][1]

        print('CV Split {} size: {}, {}'.format(i, len(X_train), len(X_test))) ##############
        if len(X_train) == 0 or len(X_test) == 0:
            for k, v in fail_trial('CV Split invalid: set size is 0', train_len=len(X_train), test_len=len(X_test)).items():
                fail_reason[k] = v
            return False

        if len(X_test) != len(test_model):
            for k, v in fail_trial('CV Split invalid: test len does not match model', test_len=len(X_test), model_len=len(test_model)).items():
                fail_reason[k] = v
            return False

        # count CV frequencies
        y_model = target_trade.iloc[test_model]
        test_counts, model_counts = {k: v for k, v in zip(*[x.tolist() for x in np.unique(y_test, return_counts=True)])}, {k: v for k, v in zip(*[x.tolist() for x in np.unique(y_model, return_counts=True)])}
        print('CV Split {} freqs: {} | Model freqs: {}'.format(i, test_counts, model_counts))
        if test_counts != model_counts:
            for k, v in fail_trial('CV Split invalid: test freqs do not match model', test_freqs=test_counts, model_freqs=model_counts).items():
                fail_reason[k] = v
            return False
        return True

    clf_params['prefit_callback'] = check_split_model
    clf_params['prefit_params'] = {'cv_model_test':cv_model}

    clf = clf_method(**clf_params)

    try:
        clf.fit(prices_indi, target_indi)
        return clf
    except Exception as e:
        traceback.print_exc()
        if len(fail_reason) > 0:
            return fail_reason
        else:
            return fail_trial('ClassifierCV error: %s'%(str(e)))

def do_cv_fit_score(prices_model, target_model, prices_indi, target_indi, prices_trade, target_trade, cv, params, clf_method, score_backtest=False, **clf_params):
    clf = do_cv_fit(prices_model=prices_model, target_model=target_model, prices_indi=prices_indi, target_indi=target_indi
                    , prices_trade=prices_trade, target_trade=target_trade, cv=cv, params=params
                    , clf_method=clf_method, **clf_params
                    )
    if isinstance(clf, dict) and clf['status'] == hp.STATUS_FAIL:
        return clf
    else:
        return do_score(clf, params, prices_indi, prices_trade, backtest=score_backtest)

def do_score(clf_cv, params, prices, prices_trade, backtest=False):
    # score
    print('Scoring') ##############
    agg_method = 'concatenate'

    acc = clf_cv.score_cv(skm.accuracy_score, aggregate=agg_method)
    precision, recall, fscore, support = clf_cv.score_cv(skm.precision_recall_fscore_support, aggregate=agg_method)
    brier = clf_cv.score_cv(skm.brier_score_loss, aggregate=agg_method, proba_positive=True)
    logloss = clf_cv.score_cv(skm.log_loss, aggregate=agg_method)

    # prep backtrader score
    end_offset = abs(params['data__params']['end_target']) #+ abs(params['data__params']['start_target'])

    y_test = clf_cv.y_true
    y_pred = clf_cv.y_pred

    try:
        start_loc = prices_trade.index.get_loc(y_test.index[0])
    except KeyError:
        start_loc = 0
    try:
        end_loc = min(prices_trade.index.get_loc(y_test.index[-1])+end_offset, len(prices_trade)-1)
    except KeyError:
        end_loc = len(prices_trade)-1

    y_prices = prices_trade.iloc[int(start_loc):int(end_loc+1),:]

    if backtest:
        pnl, trade_stats = do_backtest(y_pred, y_test, y_prices, expirebars=abs(params['data__params']['end_target'])-abs(params['data__params']['start_target']))
            # issue #16: expirebars appears to be correct, because end_target-start_target is the proper bar
            # expiry. See also SeriesStrategy, which needs to check expirebars-1 due to its counting.
    else:
        pnl, trade_stats = None, None

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
    return out

def get_best_verify_key(verify):
    if verify is None: return None
    out, out_key = None, None
    for verify_key, verify_unit in verify.items():
        if 'accuracy' not in verify_unit['pre']: continue
        if out is None or out['accuracy'] < verify_unit['pre']['accuracy']:
            out = verify_unit['pre']
            out_key = verify_key
    return out_key

def get_best_verify_accuracy(verify):
    if verify is None: return 0.
    best_key = get_best_verify_key(verify)
    if best_key is not None:
        if 'accuracy' in verify[best_key]['pre']:
            return verify[best_key]['pre']['accuracy']
    return 0.

def get_avg_verify_accuracy(verify):
    if verify is None: return 0.
    out = []
    for _, verify_unit in verify.items():
        if 'accuracy' not in verify_unit['pre']: continue
        out.append(verify_unit['pre']['accuracy'])
    return statistics.mean(out) if len(out) > 0 else 0.

def get_verify_accuracy(verify):
    return get_avg_verify_accuracy(verify)
    # return get_best_verify(verify)['accuracy']

####################################
# Data and CV
####################################

def doing_verify(cv_params):
    return len(cv_params['verify_factor']) > 0 if isinstance(cv_params['verify_factor'], Iterable) else cv_params['verify_factor'] > 0 if cv_params['verify_factor'] is not None else False

def get_verify_n(test_n, factor, inverse=False):
    if inverse:
        return test_n - math.ceil(test_n * factor)
    else:
        return math.ceil(test_n * factor)

def get_transforms(cv_params, base_only=False):
    # add master transform to end of list
    transforms = list(cv_params['transforms']) if not base_only and cv_params['transforms'] is not None and len(cv_params['transforms']) > 0 else []
    transforms.append({
        'master': True
        , 'test_size': cv_params['test_size']
        , 'test_n': cv_params['test_n']
    })
    return transforms

def get_split_sizes(transforms, verify_factors=[1], separate_verify=False):
    # todo: separate verify's test split needs to reflect start/end index;
    # train splits must be rolled into nominal train split
    total_test_size, total_verify_size, total_post_size = 0, 0, 0
    for transform in transforms:
        total_test_size += transform['test_size'] * transform['test_n']
        if 'master' in transform:
            total_verify_size += transform['test_size'] * get_verify_n(transform['test_n'], max(verify_factors))
            total_post_size += transform['test_size'] * get_verify_n(transform['test_n'], min(verify_factors), inverse=True) ### todo: clarify min/max?
        else:
            if separate_verify:
                total_verify_size += transform['test_size'] * transform['test_n']
                total_post_size += transform['test_size'] * transform['test_n']
    return total_test_size, total_verify_size, total_post_size

def get_sample_len(data_params, cv_params):
    transforms = get_transforms(cv_params)

    # get base train size
    base_train_size = cv_params['train_size']
    
    if doing_verify(cv_params):
        # add test size to nominal train size, because we're doing verification
        # make verify size the nominal test size
        verify_factors = cv_params['verify_factor'] if isinstance(cv_params['verify_factor'], Iterable) else [cv_params['verify_factor']] if cv_params['verify_factor'] is not None else [1]
        total_test_size, total_verify_size, total_post_size = get_split_sizes(transforms, verify_factors=verify_factors)
        base_train_size += total_test_size
        base_test_size = total_verify_size
        base_post_size = total_post_size
    else:
        base_test_size = cv_params['test_n'] * cv_params['test_size']
        base_post_size = 0

    return {
        'train': base_train_size
        , 'test': base_test_size
        , 'post': base_post_size
        , 'target': int(abs(data_params['end_target'])) #+ abs(data_params['start_target']))
    }

def do_data(data_params, cv_params):
    sample_len = get_sample_len(data_params, cv_params) # all needed data points for train test and target
    prices = load_prices(
        data_params['instrument']
        , data_params['granularity']
        , start_index=data_params['start_index']
        , end_index=data_params['end_index']
        , source=data_params['source']
        , sample_len=sample_len
        , dir=data_params['dir']
        , from_test=True
        , target_gap=cv_params['target_gap']
        , index_to_datetime=data_params['index_to_datetime']
        , datetime_format=data_params['datetime_format']
    )
    target = get_target(prices, data_params['end_target'], start_offset=data_params['start_target'])
    return prices, target

def get_cv(prices, data_params, cv_params, base_only=False, do_verify=False):
    if 'cv' in cv_params:
        return cv_params['cv'](**cv_params['params'])
    elif 'single_split' in cv_params:
        return SingleSplit(test_size=cv_params['single_split'])

    # else, construct chained WindowSplit CV
    # base params go last
    transforms = get_transforms(cv_params)
    master_transform = transforms[-1] # should have master in it
    sample_len = get_sample_len(data_params, cv_params)
    target_gap = cv_params['target_gap'] if 'target_gap' in cv_params else False

    verify_cv = []
    verify_factors = [1] if not doing_verify(cv_params) else cv_params['verify_factor'] if isinstance(cv_params['verify_factor'], Iterable) else [cv_params['verify_factor']] if cv_params['verify_factor'] is not None else [1]

    total_test_size, total_verify_size, total_post_size = get_split_sizes(transforms, verify_factors=verify_factors)

    # assume that data bounds accurately encompass verify_n*test_size + test_n*test_size
    data_post_end = len(prices)-data_params['end_buffer'] # exclusive post end
    data_verify_end = data_post_end-total_post_size+1 # exclusive verify end, inclusive post start
    data_test_end = data_verify_end-total_verify_size # exclusive test end, inclusive verify start
    data_train_end = data_test_end-total_test_size # exclusive train end, inclusive test start

    data_post_start = data_verify_end

    if target_gap:
        data_verify_end -= sample_len['target']
        data_test_end -= sample_len['target']
        data_train_end -= sample_len['target']

    data_verify_start = data_test_end-sum([transform['test_size']*transform['test_n'] 
                                           for transform in transforms if 'master' not in transform]
                                          )
        # this is different from verify master split, as pre-transforms must run before master split, unless separate_verify is true (todo)

    post_able = do_verify and len(prices) >= sum(sample_len.values()) + data_params['start_buffer'] + data_params['end_buffer'] - sample_len['target'] + (sample_len['target'] if target_gap else 0)
        ### todo: do per factor, not just all of them

    for verify_factor in verify_factors:
        # verify factor for verification split, verify subfactor for post split (if available)
        if post_able: ### todo: do per factor, not just all of them
            verify_subfactors = [{'post': False, 'factor': verify_factor}, {'post': True, 'factor': 1-verify_factor}]
        else:
            verify_subfactors = [{'post': False, 'factor': verify_factor}]

        verify_subcv = []
        for verify_subfactor_unit in verify_subfactors:
            factor_is_post, verify_subfactor = verify_subfactor_unit['post'], verify_subfactor_unit['factor']
            transform_cv = []
            prior_train_size = cv_params['train_size']

            if not do_verify:
                test_start = data_train_end
            else:
                if factor_is_post and target_gap:
                    test_start = data_verify_start + sample_len['target']
                else:
                    test_start = data_verify_start

            if factor_is_post:
                prior_test_size = master_transform['test_size'] * get_verify_n(master_transform['test_n'], max(verify_factors))
            else:
                prior_test_size = master_transform['test_size'] * (get_verify_n(master_transform['test_n'], max(verify_factors)) - get_verify_n(master_transform['test_n'], verify_subfactor))

            prior_data_size = cv_params['train_size']
            
            for transform in transforms:
                # Window size calculation: [train = (sum(test len) + train len)] + sum(test len)
                if not do_verify:
                    current_test_size = transform['test_size'] * transform['test_n']
                    initial_test_index = test_start + prior_test_size
                    final_index = initial_test_index + current_test_size
                    prices_size = len(prices)
                else:
                    if 'master' in transform:
                        current_test_size = transform['test_size'] * get_verify_n(transform['test_n'], verify_subfactor)
                    else:
                        current_test_size = transform['test_size'] * transform['test_n']
                    initial_test_index = test_start + prior_test_size # inclusive start of verify split
                    final_index = initial_test_index + current_test_size

                if True or 'master' in transform:
                    prices_size = len(prices)
                else:
                    # hack: CV needs to be relative to data size, which for transforms is
                    # truncated in CalibratedCV
                    # so if transform is 'master', pass the original prices size
                    # else, pass the sum of train size; all prior test sizes; and current test size
                    prior_data_size += current_test_size
                    prices_size = prior_data_size
                    prices_size_diff = len(prices)-prices_size

                train_size = prior_train_size

                if 'master' in transform:
                    args = {
                        'test_size': abs(transform['test_size'])
                        , 'step_size': abs(transform['test_size'])
                        , 'initial_test_index': initial_test_index-prices_size-1
                        , 'final_index': final_index-prices_size
                    }

                    if final_index-prices_size >= 0 and final_index-prices_size <= 1: 
                        # hack: this should only happen if we're at the last data row (e.g., factor_is_post)
                        # clear final_index so we don't erroneously clip it severely
                        args['final_index'] = None

                    if cv_params['train_sliding']:
                        args['initial_train_index'] = 0
                        if base_only and 'master' in transform: # HACK: change train length to base; all else is correct
                            args['sliding_size'] = cv_params['train_size']
                        else:
                            args['sliding_size'] = train_size
                    else:
                        if base_only and 'master' in transform: # HACK: change train length to base; all else is correct
                            args['initial_train_index'] = min(0, args['initial_test_index']-cv_params['train_size'])
                        else:
                            args['initial_train_index'] = min(0, args['initial_test_index']-train_size)
                        args['sliding_size'] = None
                else:
                    # hack: transform CVs are relative to the required size of transform
                    # because in classifyCV, data length passed is exactly what is needed for
                    # non-master transforms
                    args = {
                        'test_size': abs(transform['test_size'])
                        , 'step_size': abs(transform['test_size'])
                        , 'initial_test_index': -current_test_size
                        , 'final_index': None
                    }
                    if cv_params['train_sliding']:
                        args['initial_train_index'] = 0
                        args['sliding_size'] = train_size
                    else:
                        args['initial_train_index'] = -current_test_size-train_size
                        args['sliding_size'] = None
                
                transform_cv.append(WindowSplit(**args))
                prior_train_size += current_test_size
                prior_test_size += current_test_size
            verify_subcv.append(transform_cv)

        if not do_verify:
            verify_cv.append(verify_subcv[-1])
        else:
            verify_cv.append(verify_subcv)

    if not do_verify and len(verify_cv) > 0:
        if base_only:
            return [verify_cv[0][-1]]
        else:
            return verify_cv[0]
    else:
        if base_only:
            return [[unit[-1]] for unit in verify_cv]
        else:
            return verify_cv

####################################
# Indicators
####################################

def do_indicators(
    **indi_params_
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
        return master_union[0]
    elif len(master_union) > 1:
        return make_union(*master_union)
    else:
        return None

####################################
# Classifiers and Transforms
####################################

def do_classifier(
    cv_params, 
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
                # hack: if first transform is calibration, set objective='binary:logitraw'
                clf_inputs = {**clf_params}
                transforms = get_transforms(cv_params)
                if 'calibration' in transforms[0]:
                    clf_inputs['objective'] = 'binary:logitraw'
                pipe_pieces.append(XGBClassifier(**clf_inputs))

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

def do_classifier_transforms(base_clf, cv_list, cv_params, base_only=False, **kwargs):
    # add master transform to end of list
    transforms = get_transforms(cv_params, base_only=base_only)

    clf = clone(base_clf)
    for i, transform in enumerate(transforms):
        if 'calibration' in transform:
            clf = CalibratedClassifierCV(base_estimator=clf, method=transform['method'], cv=cv_list[i])
        elif 'threshold' in transform:
            clf = ThresholdClassifierCV(base_estimator=clf, method=transform['method'], cv=cv_list[i])
        elif 'cutoff' in transform:
            clf = CutoffClassifierCV(base_estimator=clf, cv=cv_list[i])
        elif 'master' in transform:
            clf = ClassifierCV(base_estimator=clf, cv=cv_list[i], **kwargs)
        
    return clf

def do_backtest(
    y_pred, y_true, y_prices
    , expirebars = 1
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
        , SeriesStrategy, 'signals', strategy_kwargs={'tradeintervalbars':0, 'tradeexpirebars':expirebars, 'stake':1}
        , analyzer_name=['BasicStats'], analysis_key=[[]], score_name=[None] #, score_name=['stats']
        , initial_cash=initial_balance
    )

    # get score
    results = bts._bt_score(y_pred, y_true=y_true)
    results['won'].pop('streak')
    results['lost'].pop('streak')
    pnl = results['all']['pnl']['total']

    return pnl, results

####################################
# Transforms Optimization
####################################

def do_transform_optimization(base_params, limit=0, super_threshold=0.65, super_field='accuracy'):
    if not limit: return None

    print('Finding best transform...')

    param_args = copy.deepcopy(base_params)
    param_args['cv__params'].pop('doing_transforms')
    param_args['cv__params'].pop('transforms')

    param_args['meta__params']['super_threshold'] = super_threshold
    param_args['meta__params']['super_field'] = super_field

    args = {
        'data__args': param_args['data__params']
        , 'indicator__args': param_args['indicator__params']
        , 'classifier__args': param_args['classifier__params']
        , 'cv__args': param_args['cv__params']
        , 'meta__args': param_args['meta__params']
    }
    transform_space = get_space(args=args, do_transforms=True)

    trials = Trials()
    fmin(do_candidate, space=transform_space, algo=tpe.suggest, max_evals=limit, trials=trials, return_argmin=False)
    # todo: own ctrl+c handler

    ok_count = len([t for t in trials.trials if t['result']['status'] == hp.STATUS_OK])
    if ok_count == 0:
        print('Best transform not found')
        return None

    best_result = trials.best_trial['result']

    if 'base' not in best_result or super_field not in best_result['base']:
        print('Best transform super_field %s not found'%(super_field))
        return None

    if best_result['base'][super_field] < super_threshold:
        print('Best transform not found, best score: %s' % best_result['base'][super_field])
        return None

    best_result['transform_params'] = trials.argmin

    print('Best transform:')
    pprint.pprint(best_result['transform_params'])

    return best_result

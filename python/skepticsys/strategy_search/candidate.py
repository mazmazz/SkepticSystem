import numpy as np
import pandas as pd
import hyperopt as hp
from imblearn.pipeline import make_pipeline
from collections import OrderedDict
import copy

# parent submodules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessors import IndicatorTransformer, CopyTransformer, DeltaTransformer, ShiftTransformer
from datasets import load_prices, get_target
from pipeline import make_union
sys.path.pop(0)
# end parent submodules

def do_candidate(params):
    try:
        return do_fit_predict(params)
    except Exception as e:
        # https://stackoverflow.com/a/1278740
        fname = os.path.split(sys.exc_info()[-1].tb_frame.f_code.co_filename)[1]
        msg = '%s, %s, %s | %s' % (sys.exc_info()[0].__name__, fname, sys.exc_info()[-1].tb_lineno, str(e))
        return {'status': hp.STATUS_FAIL, 'msg': 'Exception: %s'%(msg)}

def do_fit_predict(params):
    # load prices
    prices, target = do_data(**params['data__params'])
    prices_trade = prices.copy()

    # do indicators
    indi_pipeline = do_indicators(**params['indicator__params'])
    if not bool(indi_pipeline):
        return {'status': hp.STATUS_FAIL,'msg':'Indicator pipeline: No transformers'}
    prices_indi = indi_pipeline.transform(prices)

    return {'status': hp.STATUS_FAIL, 'msg': 'Finished, not implemented', 'shape': prices_indi.shape if hasattr(prices_indi,'shape') else 'No shape? ' + str(type(prices_indi))}

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

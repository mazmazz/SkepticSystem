from space import get_space
from candidate import do_candidate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from talib import MA_Type
import hyperopt.pyll.stochastic
import numpy as np
import pandas as pd
import pprint
import argparse
import os
import yaml
from io import StringIO
from time import time
import pickle
import signal
from hyperopt.mongoexp import MongoTrials

def main(args=None):
    # get space schema
    space_args = get_space_args(args)
    space = get_space(space_args)

    # get trials
    trials = load_trials(args.fmin_trials, mongo_key=args.mongo_key, reset_trials=args.fmin_reset)

    if args.show_trials:
        show_trials(trials=trials, best_params_path=args.best_params_path, best_result_path=args.best_result_path)
    elif args.fmin:
        do_fmin(space, limit=args.sample_count, trials=trials, trials_path=args.fmin_trials, best_params_path=args.best_params_path, best_result_path=args.best_result_path)
    else:
        # sample and print
        sample = get_sample(space, limit=args.sample_count, trials=trials if args.eval_trials else None, params=args.eval_params)
        if args.show_sample:
            print_sample(sample)

        # eval
        if args.eval or bool(args.eval_trials) or bool(args.eval_params):
            results = eval_sample(sample)

def load_trials(trials_path, mongo_key=None, reset_trials=False):
    is_mongo = trials_path.startswith('mongo:') if isinstance(trials_path, str) else False
    if is_mongo:
        return MongoTrials(trials_path, exp_key=mongo_key)
    else:
        if trials_path is not None and os.path.isfile(trials_path) and not reset_trials:
            return pickle.load(open(trials_path, 'rb'))
        else:
            return Trials()

def show_trials(trials, best_params_path='test_best-params.yml', best_result_path='test_best-result.yml'):
    handle_trials(trials, trials_path=None, best_params_path=best_params_path, best_result_path=best_result_path)

def get_sample(space, limit=10, trials=None, params=None):
    if trials is not None:
        ok_count = len([t for t in trials.trials if t['result']['status'] == STATUS_OK])
        if ok_count > 0:
            params = trials.argmin
        else:
            raise ValueError('Trials does not have any samples.') 
    else:
        params = load_yaml(params)

    if isinstance(params, dict) and len(params) > 0:
        print('Loaded file')
        return [space_eval(space, params)]
    else:
        print('Sampling')
        return [hyperopt.pyll.stochastic.sample(space) for i in range(limit)]

def print_sample(sample):
    if not isinstance(sample, list):
        sample = [sample]

    for i in range(len(sample)):
        print('%s | Input | %s' % (i, '='*80))
        pprint.pprint(sample[i])

def eval_sample(sample, do_print=True):
    results = []
    master_start_time = time()
    for i, sample_unit in enumerate(sample):
        start_time = time()
        result = do_candidate(sample_unit)
        end_time = time()
        results.append(result)

        if do_print:
            bench = end_time-start_time
            #print('%s | Output | %s | %02d:%08.5f' % ((i, '='*48)+divmod(bench, 60))) #mins, secs
            #pprint.pprint(result)

    if do_print:
        master_bench = time() - master_start_time
        m, s = divmod(master_bench,60)
        h, _ = divmod(m, 60)
        print('Total time: %03d:%02d:%08.5f' % (h,m,s)) #hrs, mins, secs

    return results

def do_fmin(space, trials, limit=100, trials_path='test_trials.p', best_params_path='test_best-params.yml', best_result_path='test_best-result.yml', show_trials_only=False):
    master_start_time = time()

    def sigint_handler():
        print('Interrupting...')
        handle_trials(trials, trials_path=trials_path, best_params_path=best_params_path, best_result_path=best_result_path, master_start_time=master_start_time)
        return
    
    # sigint handler to workaround scipy ctrl+c crash
    handler = handle_ctrl()

    try:
        best_params = fmin(do_candidate, space=space, algo=tpe.suggest, max_evals=limit, trials=trials)
    except KeyboardInterrupt:
        sigint_handler()
        unhandle_result = unhandle_ctrl(handler)
        return

    unhandle_result = unhandle_ctrl(handler)

    handle_trials(trials, best_params=best_params, trials_path=trials_path, best_params_path=best_params_path, best_result_path=best_result_path, master_start_time=master_start_time)

def handle_trials(trials, best_params=None, best_result=None, trials_path='test_trials.p', best_params_path='test_best-params.yml', best_result_path='test_best-result.yml', master_start_time=None):
    is_mongo = trials_path.startswith('mongo:') if isinstance(trials_path, str) else False

    if master_start_time is not None:
        master_bench = time() - master_start_time
        m, s = divmod(master_bench,60)
        h, _ = divmod(m, 60)
        print('Total time: %03d:%02d:%08.5f' % (h,m,s)) #hrs, mins, secs
    
    ok_count = len([t for t in trials.trials if t['result']['status'] == STATUS_OK])
    print('Trials count: {} ({} ok)'.format(len(trials), ok_count))
    if ok_count > 0:
        print('Lowest loss: {}'.format(min([t for t in trials.losses() if t is not None])))
        print('Average best error: {}'.format(trials.average_best_error()))
        if best_params is None:
            best_params = trials.argmin
        if best_result is None:
            best_result = trials.best_trial
            if 'result' in best_result:
                best_result = best_result['result']
                pprint.pprint(best_result)
            else:
                best_result = None

    if not is_mongo and trials_path is not None and len(trials) > 0:
        pickle.dump(trials, open(trials_path, 'wb'))

    if best_params_path is not None and best_params is not None:
        with open(best_params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)

    if best_result_path is not None and best_result is not None:
        with open(best_result_path, 'w') as f:
            f.write(pprint.pformat(best_result))

    print('Finished, saved to {}, {}, {}'.format(trials_path, best_params_path, best_result_path)) #: {}'.format(best))

def handle_ctrl(hook_sigint=None):
    if os.name != 'nt':
        import signal
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, hook_sigint)
        return original_sigint_handler
    else:
        # work around scipy's SIGINT handling: https://stackoverflow.com/a/15472811
        import imp
        import ctypes
        import _thread
        import win32api

        if hook_sigint is None:
            hook_sigint = _thread.interrupt_main

        # Load the DLL manually to ensure its handler gets
        # set before our handler.
        #basepath = imp.find_module('numpy')[1]
        #ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
        #ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))
        import numpy as np

        # Now set our handler for CTRL_C_EVENT. Other control event 
        # types will chain to the next handler.
        def handler(dwCtrlType, hook_sigint=hook_sigint):
            if dwCtrlType == 0: # CTRL_C_EVENT
                hook_sigint()
                return 1 # don't chain to the next handler
            return 0 # chain to the next handler

        win32api.SetConsoleCtrlHandler(handler, 1)
        return handler

def unhandle_ctrl(handler):
    if os.name != 'nt':
        signal.signal(signal.SIGINT, handler)
        return 1
    else:
        import win32api
        return win32api.SetConsoleCtrlHandler(handler, 0)

def get_space_args(args):
    if args.space_config is not None:
        return load_yaml(args.space_config)
    else:
        space_args = {
            'data__args': load_yaml(args.data_config) or {
                'instruments': args.data_instruments
                , 'granularities': args.data_granularities
                , 'start_index': args.data_start_index
                , 'end_index': args.data_end_index
                , 'sample_len': args.data_sample_len
                , 'source': args.data_source
                , 'dir': args.data_dir
                , 'end_target': args.data_end_target
            }
            , 'indicator__args': load_yaml(args.indicator_config) or {
                
            }
            , 'classifier__args': load_yaml(args.classifier_config) or {

            }
            , 'cv__args': load_yaml(args.cv_config) or {
                #'single_split': args.single_split
            }
        }
        return space_args

def load_yaml(value):
    if value is None: 
        return None
    elif os.path.isfile(value):
        with open(value, 'r') as fr:
            yo = yaml.load(fr)
    else: # try to interpret as string
        yo = yaml.load(StringIO(value))
    
    return yo

def str_or_none(x):
    return None if x.lower() == 'none' else str(x)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trials-path', '-tp', dest='fmin_trials', type=str_or_none, default=None
        , help='Path to file or mongo URI')
    parser.add_argument('--mongo-key', '-md', dest='mongo_key', type=str, default=None
        , help='Experiment key to read')

    parser.add_argument('--fmin', '-f', action='store_true', default=False
        , help='Run fmin optimizer')

    parser.add_argument('--eval', '-e', action='store_true', default=False
        , help='Run a randomly generated sample(s), up to --candidates.')
    parser.add_argument('--eval-trials', '-et', dest='eval_trials', action='store_true', default=False
        , help='Eval best candidate from --trials-path.')
    parser.add_argument('--eval-params', '-ep', dest='eval_params', type=str, default=None
        , help='Pre-existing params to load for eval. Can be trials pickle (.p, .pkl, .pickle) or YAML.')

    parser.add_argument('--candidates', '-s', dest='sample_count', type=int, default=1
        , help='Number of candidates to sample in --eval or --fmin.')

    parser.add_argument('--show-trials', '-st', dest='show_trials', action='store_true', default=False
        , help='Show trials statistics from --trials-path')
    parser.add_argument('--show-sample', '-ss', dest='show_sample', action='store_true', default=False
        , help='Print randomly generated or loaded sample(s)')

    parser.add_argument('--fmin-reset', '-fr', dest='fmin_reset', action='store_true', default=False
        , help='If --trials-path exists, do not load it for fmin. Has no effect if using MongoDB.')

    parser.add_argument('--save-best-params', '-sbp', dest='best_params_path', type=str_or_none, default=None
        , help='Path to save best candidate in fmin or --show-trials.')
    parser.add_argument('--save-best-result', '-sbr', dest='best_result_path', type=str_or_none, default=None
        , help='Path to save best trial result.')

    parser.add_argument('--space-config', '-sc', dest='space_config', type=str, default=None)
    parser.add_argument('--data-config', '-dc', dest='data_config', type=str, default=None)
    parser.add_argument('--indi-config', '-ic', dest='indicator_config', type=str, default=None)
    parser.add_argument('--classifier-config', '-cc', dest='classifier_config', type=str, default=None)
    parser.add_argument('--cv-config', '-vc', dest='cv_config', type=str, default=None)

    # data parameters
    parser.add_argument('--instruments', '-di', dest='data_instruments', type=str, nargs='+', default=['USDJPY'])
    parser.add_argument('--granularities', '-dg', dest='data_granularities', type=str, nargs='+', default=['H1'])
    parser.add_argument('--start-index', '-dsi', dest='data_start_index', type=int, default=None) ### TODO ### supposed to be str
    parser.add_argument('--end-index', '-dei', dest='data_end_index', type=int, default=None) ### TODO ### supposed to be str
    parser.add_argument('--sample-len', '-dsl', dest='data_sample_len', type=int, default=-12000)
    parser.add_argument('--source', '-ds', dest='data_source', type=str, default='csv')
    parser.add_argument('--dir', '-dd', dest='data_dir', type=str, default='D:\\Projects\\Prices'
        , help='Dir of CSV prices')
    parser.add_argument('--end-target', '-de', dest='data_end_target', type=int, default=-61
        , help='End offset of price target')

    # cv parameters
    #parser.add_argument('--single-split', '-sp', dest='single_split', type=int, default=None) # 100

    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

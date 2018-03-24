from optimizer.space import get_space
from optimizer.candidate import do_candidate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from talib import MA_Type
import hyperopt.pyll.stochastic
import numpy as np
import pandas as pd
import pprint
import argparse
import os
import yaml
import statistics
from io import StringIO
from time import time
import pickle
import signal
from hyperopt.mongoexp import MongoTrials
from datetime import datetime

# globals for interrupt access
Interrupt_Time = None
Interrupt_Count = 0
Super_Threshold = 0.65
Super_Field = 'accuracy'
Fmin_Trials = None
Fmin_Trials_Path=None
Fmin_Best_Params_Path=None
Fmin_Best_Result_Path=None
Fmin_Master_Start_Time=None
Fmin_Trials_N = 1

def main(args=None):
    # get space schema
    space_args = get_space_args(args)
    space = get_space(space_args)

    # get trials
    trials = load_trials(args.fmin_trials, mongo_key=args.mongo_key, reset_trials=args.fmin_reset)

    if args.show_trials:
        show_trials(trials=trials, best_params_path=args.best_params_path, best_result_path=args.best_result_path, show_n=args.trials_n, super_threshold=args.super_threshold, super_field=args.super_field)
    elif args.fmin:
        do_fmin(space, limit=args.sample_count, trials=trials, trials_path=args.fmin_trials, best_params_path=args.best_params_path, best_result_path=args.best_result_path, show_n=args.trials_n, super_threshold=args.super_threshold, super_field=args.super_field)
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

def show_trials(trials, best_params_path='test_best-params.yml', best_result_path='test_best-result.yml', show_n=1, super_threshold=0.65, super_field='accuracy'):
    handle_trials(trials, trials_path=None, best_params_path=best_params_path, best_result_path=best_result_path, show_n=show_n, super_threshold=super_threshold, super_field=super_field)

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

def do_fmin(space, trials, limit=100, trials_path='test_trials.p', best_params_path='test_best-params.yml', best_result_path='test_best-result.yml', show_n=1, show_trials_only=False, super_threshold=0.65, super_field='params'):
    global Fmin_Trials, Fmin_Trials_Path, Fmin_Best_Params_Path, Fmin_Best_Result_Path, Fmin_Master_Start_Time, Fmin_Trials_N, Super_Threshold, Super_Field
    Fmin_Trials_Path, Fmin_Best_Params_Path, Fmin_Best_Result_Path, Fmin_Trials_N = trials_path, best_params_path, best_result_path, show_n
    Super_Threshold, Super_Field = super_threshold, super_field
    
    Fmin_Master_Start_Time = time()

    int_count = 0
    int_time = None
    def sigint_handler():
        print('Interrupting...')
        handle_trials(trials, trials_path=trials_path, best_params_path=best_params_path, best_result_path=best_result_path, master_start_time=Fmin_Master_Start_Time, show_n=show_n, show_output=False)
        return
    
    # sigint handler to workaround scipy ctrl+c crash
    handler = handle_ctrl(hook_sigint=handle_staged_interrupt)

    try:
        Fmin_Trials = trials # global for interrupt access
        best_params = fmin(do_candidate, space=space, algo=tpe.suggest, max_evals=limit, trials=Fmin_Trials)
    except KeyboardInterrupt:
        sigint_handler()
        unhandle_result = unhandle_ctrl(handler)
        return

    unhandle_result = unhandle_ctrl(handler)

    handle_trials(trials, trials_path=trials_path, best_params_path=best_params_path, best_result_path=best_result_path, master_start_time=Fmin_Master_Start_Time, show_n=show_n)

def handle_trials(trials, trials_path=None, best_params_path=None, best_result_path=None, master_start_time=None, show_n=1, show_output=True, super_threshold=0.65, super_field='accuracy'):
    is_mongo = trials_path.startswith('mongo:') if isinstance(trials_path, str) else False

    if master_start_time is not None:
        master_bench = time() - master_start_time
        m, s = divmod(master_bench,60)
        h, _ = divmod(m, 60)
        print('Total time: %03d:%02d:%08.5f' % (h,m,s)) #hrs, mins, secs
    
    ok_count = len([t for t in trials.trials if t['result']['status'] == STATUS_OK])
    print('Trials count: {} ({} ok)'.format(len(trials), ok_count))

    if ok_count > 0:
        best_trials = get_best_trials(trials, show_n=show_n, get_lowest_loss=True, super_threshold=super_threshold, super_field=super_field)
        for i, best_trial in enumerate(best_trials):
            if show_output: print('='*5 + ' Trial {} '.format(i) + '='*5)
            if show_output: print('Loss: {}'.format(best_trial['result']['loss'] if best_trial is not None else None))
            
            best_params = get_trial_args(best_trial)
            best_result = get_trial_result(best_trial)
            
            if best_result is not None:
                if show_output: pprint.pprint(best_result)

            if best_params_path is not None and best_params is not None:
                if show_n > 1:
                    split_path = os.path.splitext(best_params_path)
                    path = '{}-t{}{}'.format(split_path[0], i, split_path[1])
                else:
                    path = best_params_path
                with open(path, 'w') as f:
                    yaml.dump(best_params, f, default_flow_style=False)

            if best_result_path is not None and best_result is not None:
                if show_n > 1:
                    split_path = os.path.splitext(best_result_path)
                    path = '{}-t{}{}'.format(split_path[0], i, split_path[1])
                else:
                    path = best_result_path
                with open(path, 'w') as f:
                    f.write(pprint.pformat(best_result))

    if not is_mongo and trials_path is not None and len(trials) > 0:
        pickle.dump(trials, open(trials_path, 'wb'))

    print('Saved to {}, {}, {}'.format(trials_path, best_params_path, best_result_path)) #: {}'.format(best))

def get_best_trials(trials, show_n=1, get_lowest_loss=False, super_threshold=0.65, super_field='accuracy'):
    ok_count = len([t for t in trials.trials if t['result']['status'] == STATUS_OK])
    if ok_count == 0:
        return []
    out = [trials.best_trial] if get_lowest_loss else []

    # hack: hardcoded best trial requirements, will certainly be changed
    try:
        candidates = [t for t in trials.trials
                      if t['result']['status'] == STATUS_OK
                      ]
        finalists = [c for c in candidates 
                     if 'super_score' in c['result'] and c['result']['super_score'] is not None
                     and c['result']['super_score'] > super_threshold
                     ]

        if len(finalists) == 0:
            print('No candidates obtained for score criteria')
            return out
        else:
            print('Trials meeting score criteria: %s'%len(finalists))

        # ranking https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
        # combine multiple ranks (rank aggregation): https://stats.stackexchange.com/questions/56852/overall-rank-from-multiple-ranked-lists
        # weight accuracy diff by 1.5, accuracy base by 1.0
        acc_base = np.array([c['result']['base']['accuracy'] for c in finalists])
        acc_base_rank = (-acc_base).argsort().argsort()+1.
        # acc_verify = np.array([c['result']['verify_avg_accuracy'] for c in finalists])
        # acc_verify_rank = (-acc_verify).argsort().argsort()+1.
        acc_diffs = np.array([abs(c['result']['base']['accuracy']-c['result']['verify_avg_accuracy']) for c in finalists])
        acc_diffs_rank = acc_diffs.argsort().argsort()+1.

        final_rank = (((acc_base_rank+acc_diffs_rank*1.5)/2).argsort().argsort()+1.).tolist()

        for i in range(show_n):
            if i >= len(finalists): break
            out.append(finalists[np.argmin(final_rank)])
            final_rank[np.argmin(final_rank)] = np.inf
    except Exception as e:
        print('Error obtaining best trial: {}'.format(e))
    
    return out

def get_trial_args(trial):
    if trial is None: return None
    vals = trial['misc']['vals']
    # unpack the one-element lists to values
    # and skip over the 0-element lists
    rval = {}
    for k, v in list(vals.items()):
        if v:
            rval[k] = v[0]
    return rval

def get_trial_result(trial):
    if trial is None: return None
    elif 'result' in trial: return trial['result']
    else: return None

def handle_staged_interrupt():
    global Interrupt_Time, Interrupt_Count, Super_Threshold, Super_Field, Fmin_Trials, Fmin_Trials_Path, Fmin_Best_Params_Path, Fmin_Best_Result_Path, Fmin_Master_Start_Time, Fmin_Trials_N

    print('\n' + '='*20 + '\n')

    # interrupt 3 times in 5 seconds = quit
    # else, print trial results
    if Interrupt_Time is None or (time()-Interrupt_Time) > 5:
        Interrupt_Time = time()
        Interrupt_Count = 0

    Interrupt_Count += 1

    if Interrupt_Count > 2:
        Interrupt_Time = None
        Interrupt_Count = 0
        import _thread
        _thread.interrupt_main()
    else:
        if Interrupt_Count == 1: # first signal only
            handle_trials(Fmin_Trials, trials_path=Fmin_Trials_Path, best_params_path=Fmin_Best_Params_Path, best_result_path=Fmin_Best_Result_Path, master_start_time=Fmin_Master_Start_Time, show_n=Fmin_Trials_N, super_threshold=Super_Threshold, super_field=Super_Field) # print only
        print('Interrupt %s more times in %.1f seconds to quit' % (3-Interrupt_Count, 5-(time()-Interrupt_Time)))
        print('\n' + '='*20 + '\n')

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
            'test__args': load_yaml(args.test_config) or {
                'start_index': format_index_value(args.test_start_index, not args.no_datetime_convert, args.datetime_format)
                , 'end_index': format_index_value(args.test_end_index, not args.no_datetime_convert, args.datetime_format)
                , 'end_target': args.test_end_target
                , 'test_size': args.test_size
                , 'test_n': args.test_n
                , 'train_size': args.train_size
                , 'train_sliding': args.train_sliding
            }
            , 'data__args': load_yaml(args.data_config) or {
                'instruments': args.data_instruments
                , 'granularities': args.data_granularities
                , 'source': args.data_source
                , 'dir': args.data_dir
                , 'datetime_convert': not args.no_datetime_convert
                , 'datetime_format': args.datetime_format
            }
            , 'indicator__args': load_yaml(args.indicator_config) or {
                
            }
            , 'classifier__args': load_yaml(args.classifier_config) or {

            }
            , 'cv__args': load_yaml(args.cv_config) or {
                #'single_split': args.single_split
            }
            , 'meta__args': load_yaml(args.meta_config) or {
                'super_threshold': args.super_threshold
                , 'super_field': args.super_field
                , 'transform_limit': args.transform_limit
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

def format_index_value(x, convert_datetime=True, datetime_format='%Y%m%d%H%M'):
    if x is None:
        return None
    elif convert_datetime:
        return datetime.strptime(x, datetime_format)
    else:
        return int(x) # \todo this should be str, don't know if rest of program supports it

def get_args():
    parser = argparse.ArgumentParser()
    group_main = parser.add_argument_group('main', 'Trials data I/O')
    group_opt = parser.add_argument_group('optimization', 'Shared optimization parameters')
    group_fmin = parser.add_argument_group('fmin')
    group_eval = parser.add_argument_group('eval')
    group_config = parser.add_argument_group('config', 'YAML files for each search space')
    group_data = parser.add_argument_group('data', 'Price data parameters')
    group_test = parser.add_argument_group('test', 'Test parameters')

    group_main.add_argument('--trials-path', '-tp', dest='fmin_trials', type=str_or_none, default=None
        , help='Path to file or mongo URI')
    group_main.add_argument('--mongo-key', '-md', dest='mongo_key', type=str, default=None
        , help='Experiment key to read')
    group_main.add_argument('--show-trials', '-st', dest='show_trials', action='store_true', default=False
        , help='Show trials statistics from --trials-path')
    group_main.add_argument('--save-best-params', '-sbp', dest='best_params_path', type=str_or_none, default=None
        , help='Path to save best candidate in fmin or --show-trials.')
    group_main.add_argument('--save-best-result', '-sbr', dest='best_result_path', type=str_or_none, default=None
        , help='Path to save best trial result.')
    group_main.add_argument('--show-sample', '-ss', dest='show_sample', action='store_true', default=False
        , help='Print randomly generated or loaded sample(s)')

    group_opt.add_argument('--candidates', '-s', dest='sample_count', type=int, default=1
        , help='Number of candidates to sample in --eval or --fmin.')
    group_opt.add_argument('--trials-n', '-tn', dest='trials_n', type=int, default=1
        , help='# of trials to show for --show-trials or fmin results. default: 1')

    group_fmin.add_argument('--fmin', '-f', action='store_true', default=False
        , help='Run fmin optimizer')
    group_fmin.add_argument('--fmin-reset', '-fr', dest='fmin_reset', action='store_true', default=False
        , help='If --trials-path exists, do not load it for fmin. Has no effect if using MongoDB.')

    group_eval.add_argument('--eval', '-e', action='store_true', default=False
        , help='Run a randomly generated sample(s), up to --candidates.')
    group_eval.add_argument('--eval-trials', '-et', dest='eval_trials', action='store_true', default=False
        , help='Eval best candidate from --trials-path.')
    group_eval.add_argument('--eval-params', '-ep', dest='eval_params', type=str, default=None
        , help='Pre-existing params to load for eval. Can be trials pickle (.p, .pkl, .pickle) or YAML.')

    group_config.add_argument('--space-config', '-sc', dest='space_config', type=str, default=None)
    group_config.add_argument('--test-config', '-tc', dest='test_config', type=str, default=None)
    group_config.add_argument('--data-config', '-dc', dest='data_config', type=str, default=None)
    group_config.add_argument('--indi-config', '-ic', dest='indicator_config', type=str, default=None)
    group_config.add_argument('--classifier-config', '-cc', dest='classifier_config', type=str, default=None)
    group_config.add_argument('--cv-config', '-vc', dest='cv_config', type=str, default=None)
    group_config.add_argument('--meta-config', '-mc', dest='meta_config', type=str, default=None)

    # data parameters
    group_data.add_argument('--instruments', '-di', dest='data_instruments', type=str, nargs='+', default=['USDJPY'])
    group_data.add_argument('--granularities', '-dg', dest='data_granularities', type=str, nargs='+', default=['H1'])
    group_data.add_argument('--source', '-ds', dest='data_source', type=str, default='csv')
    group_data.add_argument('--dir', '-dd', dest='data_dir', type=str, default='D:\\Projects\\Prices'
        , help='Dir of CSV prices')
    group_data.add_argument('--no-datetime-convert', '-dtc', dest='no_datetime_convert', action='store_true', default=False
        , help="Don't convert CSV indexes to datetime. Default: convert to datetime")
    group_data.add_argument('--datetime-format', '-dtf', dest='datetime_format', type=str, default='%Y%m%d%H%M'
        , help='Datetime format to parse from string. Default: %%Y%%m%%d%%H%%M (201701031800)')

    # test parameters
    group_test.add_argument('--start-index', '-tsi', dest='test_start_index', type=str, default=None
        , help='Index to start evaluation set. If converting to datetime (see above params), format must be in --datetime-format.')
    group_test.add_argument('--end-index', '-tei', dest='test_end_index', type=str, default=None
        , help='Index to end evaluation set. If converting to datetime (see above params), format must be in --datetime-format.')
    group_test.add_argument('--end-target', '-tt', dest='test_end_target', type=int, default=None # -61
        , help='End offset of price target')
    group_test.add_argument('--test-size', '-tss', dest='test_size', type=int, default=None) # 120
    group_test.add_argument('--test-n', '-tsn', dest='test_n', type=int, default=None) # 4
    group_test.add_argument('--train-size', '-tns', dest='train_size', type=int, default=None) # 6000
    group_test.add_argument('--train-sliding', '-tnl', dest='train_sliding', type=bool, default=None) # True
    group_test.add_argument('--super-threshold', '-sh', dest='super_threshold', type=float, default=0.65)
    group_test.add_argument('--super-field', '-sf', dest='super_field', type=str, default='accuracy')
    group_test.add_argument('--transform-limit', '-tl', dest='transform_limit', type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

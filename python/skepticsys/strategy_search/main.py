from space import get_space
from candidate import do_candidate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from talib import MA_Type
import hyperopt.pyll.stochastic
import numpy as np
import pandas as pd
import pprint
import argparse
import os
import yaml
from io import StringIO

def main(args=None):
    # get space schema
    space_args = get_space_args(args)
    space = get_space(space_args)

    # sample and print
    sample = get_sample(space, limit=args.sample_count)
    if not args.suppress_sample:
        print_sample(sample)

    # eval
    if args.eval:
        results = eval_sample(sample)

def get_sample(space, limit=10):
    return [hyperopt.pyll.stochastic.sample(space) for i in range(limit)]

def print_sample(sample):
    if not isinstance(sample, list):
        sample = [sample]

    for i in range(len(sample)):
        print('%s | Input | %s' % (i, '='*80))
        pprint.pprint(sample[i])

def eval_sample(sample, do_print=True):
    results = []
    for i, sample_unit in enumerate(sample):
        result = do_candidate(sample_unit)
        results.append(result)
        if do_print:
            print('%s | Output | %s' % (i, '='*48))
            pprint.pprint(result)

    return results

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
        }
        return space_args

def load_yaml(value):
    if value is None: return None
    elif os.path.isfile(value):
        with open(value, 'r') as fr:
            yo = yaml.load(fr)
    else: # try to interpret as string
        yo = yaml.load(StringIO(value))
    
    return yo

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--suppress-sample', '-ss', dest='suppress_sample', action='store_true', default=False)

    parser.add_argument('--sample', '-s', dest='sample_count', type=int, default=1)
    parser.add_argument('--eval', '-e', action='store_true', default=False)

    parser.add_argument('--space-config', '-sc', dest='space_config', type=str, default=None)
    parser.add_argument('--data-config', '-dc', dest='data_config', type=str, default=None)
    parser.add_argument('--indi-config', '-ic', dest='indicator_config', type=str, default=None)

    # data parameters
    parser.add_argument('--instruments', '-di', dest='data_instruments', type=str, nargs='+', default=['USDJPY'])
    parser.add_argument('--granularities', '-dg', dest='data_granularities', type=str, nargs='+', default=['H1'])
    parser.add_argument('--start-index', '-dsi', dest='data_start_index', type=str, default=None)
    parser.add_argument('--end-index', '-dei', dest='data_end_index', type=str, default=None)
    parser.add_argument('--sample-len', '-dsl', dest='data_sample_len', type=int, default=-12000)
    parser.add_argument('--source', '-ds', dest='data_source', type=str, default='csv')
    parser.add_argument('--dir', '-dd', dest='data_dir', type=str, default='D:\\Projects\\Prices'
        , help='Dir of CSV prices')
    parser.add_argument('--end-target', '-de', dest='data_end_target', type=int, default=-61
        , help='End offset of price target')

    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

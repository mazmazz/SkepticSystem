from space import get_space
from candidate import do_candidate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from talib import MA_Type
import hyperopt.pyll.stochastic
import numpy as np
import pandas as pd
import pprint
import argparse

def main(args):
    space_args = get_space_args(args)
    space = get_space(space_args)
    sample = hyperopt.pyll.stochastic.sample(space)
    do_candidate(sample)
    # print_space_sample(space, 100)

def print_space_sample(space, limit=10):
    for i in range(0,limit):
        print('%s | %s' % (i, '='*80))
        pprint.pprint(hyperopt.pyll.stochastic.sample(space))

def get_space_args(args):
    if args.space_config is not None:
        return load_yaml(args.space_config)
    else:
        space_args = {
            'data__args': {
                'instruments': args.data_instruments
                , 'granularities': args.data_granularities
                , 'start_index': args.data_start_index
                , 'end_index': args.data_end_index
                , 'sample_len': args.data_sample_len
                , 'source': args.data_source
                , 'dir': args.data_dir
                , 'end_target': args.data_end_target
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

    parser.add_argument('--space-config', '-sc', dest='space_config', type=str, default=None)

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

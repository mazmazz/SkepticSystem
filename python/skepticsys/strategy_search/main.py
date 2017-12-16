from space import get_space
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from talib import MA_Type
import hyperopt.pyll.stochastic
import numpy as np
import pandas as pd
import pprint

def main():
    space = get_space()
    print_space_sample(space, 100)

def print_space_sample(space, limit=100):
    for i in range(0,limit):
        print('%s | %s' % (i, '='*80))
        pprint.pprint(hyperopt.pyll.stochastic.sample(space))

if __name__ == '__main__':
    main()

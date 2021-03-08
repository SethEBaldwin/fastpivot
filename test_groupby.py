import pivot
import groupby
import pandas as pd
import numpy as np
import time

# N_ROWS = 4
# N_COLS = 2
# N_IDX = 2

N_ROWS = 1000000
N_COLS = 100
N_IDX = 10000

# N_ROWS = 1000000
# N_COLS = 1000
# N_IDX = 10000

NAME_IDX = 'to_be_idx'
NAME_COL = 'col1'
NAME_VALUE = 'col2'

def gen_df():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = [x for x in np.random.normal(size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_COL] = df[NAME_COL].astype(np.float64)
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # print(df)

    return df

def test_groupby_sum():

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    groupby_cython = groupby.groupby(df, by=NAME_IDX, agg='sum')
    print(msg, time.perf_counter() - tick)
    # print(groupby_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    groupby_pandas = df.groupby(by=NAME_IDX).sum()
    print(msg, time.perf_counter() - tick)
    # print(groupby_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(groupby_cython.to_numpy() - groupby_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (groupby_cython.to_numpy() == groupby_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)

    assert is_equal
    assert within_epsilon

def test_groupby_mean():

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    groupby_cython = groupby.groupby(df, by=NAME_IDX, agg='mean')
    print(msg, time.perf_counter() - tick)
    # print(groupby_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    groupby_pandas = df.groupby(by=NAME_IDX).mean()
    print(msg, time.perf_counter() - tick)
    # print(groupby_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(groupby_cython.to_numpy() - groupby_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (groupby_cython.to_numpy() == groupby_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)

    assert is_equal
    assert within_epsilon
import pivot
import pandas as pd
import numpy as np
import time
import datetime
from pivot_sparse import pivot_sparse

# N_ROWS = 1000000
# N_COLS = 100
# N_IDX = 10000

# slower here for single col, idx. faster for double
# N_ROWS = 1000000
# N_COLS = 1000  # note: pandas can't handle 10000 or even 1000... but this pivot can
# N_IDX = 100

# N_ROWS = 1000000
# N_COLS = 10
# N_IDX = 10

# These values cause memory error (out of memory)
# N_ROWS = 1000000
# N_COLS = 1000
# N_IDX = 10000

# good speed ups for these parameters
N_ROWS = 100000
N_COLS = 1000
N_IDX = 1000

# N_ROWS = 2000000
# N_COLS = 1000
# N_IDX = 50000

NAME_IDX = 'to_be_idx'
NAME_IDX2 = 'to_be_idx2'
NAME_COL = 'to_be_col'
NAME_COL2 = 'to_be_col2'
NAME_VALUE = 'value'
NAME_VALUE2 = 'value2'

def gen_df():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # print(df)

    return df

def gen_df_int():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.randint(-10, 10, size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.int64)

    # print(df)

    return df

def gen_df_multiple_values():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]
    col4 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3, col4])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE, NAME_VALUE2], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)
    df[NAME_VALUE2] = df[NAME_VALUE2].astype(np.float64)

    # print(df)

    return df

def gen_df_multiple_columns():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col_x{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col4 = ['col_y{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col4, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_COL2, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # print(df)

    return df

def gen_df_multiple_index():
    col1 = ['idx_x{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['ind_x{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col4 = ['col_y{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col4, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_IDX2, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # print(df)

    return df

def test_pivot_sum():

    print()
    print('test pivot sum')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    try:
        msg = 'sparse'
        tick = time.perf_counter()
        pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
        stime = time.perf_counter() - tick
    except:
        stime = None

    return ctime, ptime, stime

def test_pivot_mean():

    print()
    print('test pivot mean')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='mean')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='mean')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_std():

    print()
    print('test pivot std')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='std')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='std')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_max():

    print()
    print('test pivot max')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='max')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='max')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_min():

    print()
    print('test pivot min')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='min')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='min')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_count():

    print()
    print('test pivot count')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='count')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='count')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_nunique_int():
    #TODO: better test (with actual nunique not equal to counts, and longer vectors per (i, j) pair)

    print()
    print('test pivot nunique int')

    df = gen_df_int()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='nunique')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='nunique')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_pivot_median():

    print()
    print('test pivot median')

    df = gen_df()

    # time

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='median')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='median')
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    stime = None

    return ctime, ptime, stime

def test_multiple_columns():

    print()
    print('test pivot sum with multiple columns')

    df = gen_df_multiple_columns()

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        #pivot_pandas = df.pivot_table(index=[NAME_COL, NAME_COL2], columns=NAME_IDX, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        #pivot_pandas = pivot_pandas.transpose()
        ptime = time.perf_counter() - tick
    except:
        ptime = None

    try:
        msg = 'sparse'
        tick = time.perf_counter()
        pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0)
        stime = time.perf_counter() - tick
    except:
        stime = None

    return ctime, ptime, stime

def test_multiple_index():

    print()
    print('test pivot sum with multiple index')

    df = gen_df_multiple_index()

    try:
        msg = 'cython'
        tick = time.perf_counter()
        pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        ctime = time.perf_counter() - tick
    except:
        ctime = None

    try:
        msg = 'pandas'
        tick = time.perf_counter()
        pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
        ptime = time.perf_counter() - tick
    except:
        ptime = None
    
    try:
        msg = 'sparse'
        tick = time.perf_counter()
        pivot_sparse_df = pivot_sparse(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
        stime = time.perf_counter() - tick
    except:
        stime = None

    return ctime, ptime, stime


tuples = []
tuples.append(test_pivot_sum())
tuples.append(test_pivot_mean())
tuples.append(test_pivot_std())
tuples.append(test_pivot_max())
tuples.append(test_pivot_min())
tuples.append(test_pivot_count())
tuples.append(test_pivot_nunique_int())
tuples.append(test_pivot_median())
tuples.append(test_multiple_columns())
tuples.append(test_multiple_index())
index=['sum', 'mean', 'std', 'max', 'min', 'count', 'nunique', 'median', 'multicol sum', 'multiidx sum']
benchmarks_df = pd.DataFrame(tuples, columns=['fastpivot.pivot_table', 'pandas.pivot_table', 'fastpivot.pivot_sparse'], index=index)
print(benchmarks_df)
import pandas as pd
import numpy as np
import time
import fastpivot.pivot as pivot
from fastpivot.pivot_sparse import pivot_sparse

# N_ROWS = 4
# N_COLS = 2
# N_IDX = 2

# N_ROWS = 4
# N_COLS = 1
# N_IDX = 1

# N_ROWS = 1000000
# N_COLS = 100
# N_IDX = 10000

# slower here for single col, idx. faster for double
# N_ROWS = 1000000
# N_COLS = 500  # note: pandas can't handle 10000 or even 1000... but this pivot can
# N_IDX = 100

# N_ROWS = 1000000
# N_COLS = 10
# N_IDX = 10

# N_ROWS = 100000
# N_COLS = 1000
# N_IDX = 1000

# N_ROWS = 100000
# N_COLS = 100
# N_IDX = 100

# These values cause memory error (out of memory)
N_ROWS = 1000000
N_COLS = 1000
N_IDX = 10000

# N_ROWS = 10000
# N_COLS = 100
# N_IDX = 100

# N_ROWS = 100000
# N_COLS = 1000
# N_IDX = 1000

# N_ROWS = 2000000
# N_COLS = 1000
# N_IDX = 50000

# N_ROWS = 1000000
# N_COLS = 2000
# N_IDX = 50000

NAME_IDX = 'to_be_idx'
NAME_IDX2 = 'to_be_idx2'
NAME_COL = 'to_be_col'
NAME_COL2 = 'to_be_col2'
NAME_VALUE = 'value'
NAME_VALUE2 = 'value2'

print()
print('n_rows: {}'.format(N_ROWS))
print('n_columns: {}'.format(N_COLS))
print('n_idx: {}'.format(N_IDX))

def gen_df():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

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

def test_pivot_sum_aspdfalse():

    print()
    print('test pivot sum as_pd=False')

    df = gen_df()

    # time

    msg = 'sparse'
    tick = time.perf_counter()
    coo, idx_arr_unique, col_arr_unique = pivot_sparse(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, as_pd=False)
    print(msg, time.perf_counter() - tick)
    #print(coo)
    #print(idx_arr_unique)
    #print(col_arr_unique)

def test_pivot_sum():

    print()
    print('test pivot sum')

    df = gen_df()

    # time

    msg = 'sparse'
    tick = time.perf_counter()
    pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
    print(msg, time.perf_counter() - tick)
    # print(pivot_sparse_df)

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_sparse_df.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_sparse_df.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    # assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_pivot_sum_nofill():

    print()
    print('test pivot sum nofill')

    df = gen_df()

    # time

    msg = 'sparse'
    tick = time.perf_counter()
    pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None)
    print(msg, time.perf_counter() - tick)
    # print(pivot_sparse_df)

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_sparse_numpy = pivot_sparse_df.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_sparse_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_sparse_numpy = np.nan_to_num(pivot_sparse_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_sparse_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)

    # assert same_nan
    # assert within_epsilon
    is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    # assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

# def test_multiple_columns():

#     print()
#     print('test pivot sum with multiple columns')

#     df = gen_df_multiple_columns()

#     msg = 'sparse'
#     tick = time.perf_counter()
#     pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_sparse_df)

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=True)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     epsilon = 1e-8
#     within_epsilon = (np.absolute(pivot_sparse_df.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
#     print('componentwise within {} :'.format(epsilon), within_epsilon)
#     is_equal = (pivot_sparse_df.to_numpy() == pivot_pandas.to_numpy()).all()
#     print('componentwise equal: ', is_equal)
#     is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     # assert within_epsilon
#     # assert is_equal
#     # assert is_equal_pd

# def test_multiple_columns_nan():

#     print()
#     print('test pivot sum with multiple columns nan')

#     df = gen_df_multiple_columns()
#     df[NAME_COL][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan
#     df[NAME_COL2][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan

#     msg = 'sparse'
#     tick = time.perf_counter()
#     pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_sparse_df)

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     epsilon = 1e-8
#     within_epsilon = (np.absolute(pivot_sparse_df.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
#     print('componentwise within {} :'.format(epsilon), within_epsilon)
#     is_equal = (pivot_sparse_df.to_numpy() == pivot_pandas.to_numpy()).all()
#     print('componentwise equal: ', is_equal)
#     is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     # assert within_epsilon
#     # assert is_equal
#     # assert is_equal_pd

# def test_multiple_index():

#     print()
#     print('test pivot sum with multiple index')

#     df = gen_df_multiple_index()

#     msg = 'sparse'
#     tick = time.perf_counter()
#     pivot_sparse_df = pivot_sparse(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_sparse_df)

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     epsilon = 1e-8
#     within_epsilon = (np.absolute(pivot_sparse_df.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
#     print('componentwise within {} :'.format(epsilon), within_epsilon)
#     is_equal = (pivot_sparse_df.to_numpy() == pivot_pandas.to_numpy()).all()
#     print('componentwise equal: ', is_equal)
#     is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     # assert within_epsilon
#     # assert is_equal
#     # assert is_equal_pd

# def test_multiple_index_nan():

#     print()
#     print('test pivot sum with multiple index nan')

#     df = gen_df_multiple_index()
#     df[NAME_IDX][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan
#     df[NAME_IDX2][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan

#     msg = 'sparse'
#     tick = time.perf_counter()
#     pivot_sparse_df = pivot_sparse(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_sparse_df)

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     epsilon = 1e-8
#     within_epsilon = (np.absolute(pivot_sparse_df.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
#     print('componentwise within {} :'.format(epsilon), within_epsilon)
#     is_equal = (pivot_sparse_df.to_numpy() == pivot_pandas.to_numpy()).all()
#     print('componentwise equal: ', is_equal)
#     is_equal_pd = pivot_sparse_df.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     # assert within_epsilon
#     # assert is_equal
#     # assert is_equal_pd

# def contrived():
#     col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
#     col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
#     col3 = [x for x in np.random.normal(size=N_ROWS)]

#     data = np.transpose([col1, col1, col2, col3])
#     df = pd.DataFrame(data, columns=[NAME_IDX, NAME_IDX2, NAME_COL, NAME_VALUE], index=range(len(data)))
#     df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

#     print(df)

#     msg = 'sparse'
#     tick = time.perf_counter()
#     pivot_sparse_df = pivot_sparse(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_sparse_df)

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_pandas)

#test_pivot_sum_aspdfalse()
#test_pivot_sum()
import pandas as pd
import numpy as np
import time
import datetime
import fastpivot.pivot as pivot

# NOTE on speed: 
# this pivot tends to be faster than pandas when N_ROWS, N_COLS and N_IDX are large
# this pivot tends to be slightly faster than pandas with single idx and col and with N_COLS and N_IDX small
# this pivot tends to be slower than pandas with multiple idx or col and with N_ROWS is large and N_COLS, N_IDX small

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

# N_ROWS = 10000
# N_COLS = 100
# N_IDX = 100

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

def test_pivot_median_int():

    print()
    print('test pivot median int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd

def test_pivot_nan_index_dropnacolidx():

    print()
    print('test pivot nan index dropna_colidx=False')

    df = gen_df()

    df[NAME_IDX][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna_idxcol=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

def test_pivot_multiple_values_string_nunique_nan():

    print()
    print('test pivot multiple values string nunique_nan')

    df = gen_df_multiple_columns()
    df[NAME_COL2][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_COL2, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=NAME_COL, values=NAME_COL2, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    #assert is_equal
    #assert is_equal_pd

def test_pivot_nan_column_dropnacolidx():

    print()
    print('test pivot nan column dropna_colidx=False')

    df = gen_df()

    df[NAME_COL][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna_idxcol=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

def test_pivot_nan_column_nodrop():

    print()
    print('test pivot nan column nodrop')

    df = gen_df()

    df[NAME_COL][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_datetime():
    # pandas fills sum with 0.0 automatically? huh? that is silly. what is going on?

    print()
    print('test pivot datetime')

    col1 = [x for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in np.random.choice(a=['2016-10-28', '2016-11-04', '2016-12-23', '2017-01-15', '2017-02-05', '2017-03-26'], size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_IDX] = df[NAME_IDX].astype('category')
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # codes, uniques = df[NAME_IDX].factorize(sort=True)
    # codes2, uniques2 = df[NAME_COL].factorize(sort=True)
    # first_col_nans = set(range(N_IDX)) - {x[0] for x in zip(codes, codes2) if uniques2[x[1]] == datetime.datetime.strptime('2016-10-28', '%Y-%m-%d')}
    # first_col_nans_list = sorted(list(first_col_nans))
    # print(first_col_nans_list)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)
    # pivot_cython.info()
    # print(pivot_cython.loc[first_col_nans_list])

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)
    # pivot_pandas.info()
    # print(pivot_pandas.loc[first_col_nans_list])

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_date():

    print()
    print('test pivot date')

    col1 = [x for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in np.random.choice(a=['2016-10-28', '2016-11-04', '2016-12-23', '2017-01-15', '2017-02-05', '2017-03-26'], size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_IDX] = df[NAME_IDX].astype('category')
    df[NAME_COL] = df[NAME_COL].dt.date
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)
    # pivot_cython.info()

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)
    # pivot_pandas.info()

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_cat_bool():

    print()
    print('test pivot cat bool')

    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = [x for x in np.random.choice(a=[False, True], size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_IDX] = df[NAME_IDX].astype('category')
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)
    # pivot_cython.info()

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)
    # pivot_pandas.info()

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_nunique_fillNone():
    #TODO: better test (with actual nunique not equal to counts, and longer vectors per (i, j) pair)

    print()
    print('test pivot nunique fill none')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_nan_value():

    print()
    print('test pivot nan value')

    df = gen_df()

    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan
    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_count_fillNone():

    print()
    print('test pivot count fill None')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_count_fillNone_str():

    print()
    print('test pivot count fill None with str')

    df = gen_df_multiple_columns()
    df[NAME_COL2][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_COL2, fill_value=None, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_COL2, fill_value=None, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_nan_value_fillna0():

    print()
    print('test pivot nan value fillna=0')

    df = gen_df()

    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan
    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_nan_index():

    print()
    print('test pivot nan index')

    df = gen_df()

    df[NAME_IDX][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_nan_column():

    print()
    print('test pivot nan column')

    df = gen_df()

    df[NAME_COL][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # print(df)

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_values_list():
    # inexplicably, pandas does not sort here.
    # that would be fine if they didn't sort aggfunc and values in all other cases...
    # this pivot will sort in all cases

    print()
    print('test pivot values list')

    df = gen_df()

    aggfunc_list = ['median', 'sum']

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc=aggfunc_list)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc=aggfunc_list)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_values_list_nan():
    # inexplicably, pandas does not sort here.
    # that would be fine if they didn't sort aggfunc and values in all other cases...
    # this pivot will sort in all cases

    print()
    print('test pivot values list nan')

    df = gen_df()
    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    aggfunc_list = ['max', 'mean']

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc=aggfunc_list)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc=aggfunc_list)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

# def test_pivot_multiple_values_list():

#     print()
#     print('test pivot multiple values list')

#     df = gen_df_multiple_columns()

#     aggfunc_dict = {NAME_COL2: 'count', NAME_VALUE: ['median', 'sum']}

#     # time

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_pandas)

#     # check results are equal

#     is_equal_pd = pivot_cython.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     assert is_equal_pd

# def test_pivot_multiple_values_list_nan():

#     print()
#     print('test pivot multiple values list nan')

#     df = gen_df_multiple_columns()
#     df[NAME_COL2][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan
#     df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

#     aggfunc_dict = {NAME_COL2: 'count', NAME_VALUE: ['min', 'median']}

#     # time

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     print(pivot_pandas)

#     # check results are equal

#     is_equal_pd = pivot_cython.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     assert is_equal_pd

def test_pivot_sum():

    print()
    print('test pivot sum')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

    # compare to groupby hack

    # msg = 'pandas groupby'
    # tick = time.perf_counter()
    # groupby_pandas = df.groupby([NAME_COL, NAME_IDX])[NAME_VALUE].sum().unstack(level=NAME_COL).fillna(0)
    # print(msg, time.perf_counter() - tick)
    # # print(groupby_pandas)

    # assert (groupby_pandas.equals(pivot_pandas))


def test_pivot_sum_fillnan():

    print()
    print('test pivot sum fill nan')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=np.nan, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=np.nan, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

# def test_pivot_sum_silly():

#     print()
#     print('test pivot sum with index, columns list of single string')

#     df = gen_df()

#     # time

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=[NAME_IDX], columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
#     print('componentwise equal: ', is_equal)
#     epsilon = 1e-8
#     within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
#     print('componentwise within {} :'.format(epsilon), within_epsilon)
#     is_equal_pd = pivot_cython.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     assert within_epsilon
#     assert is_equal
#     assert is_equal_pd

# def test_pivot_multiple_values_string():

#     print()
#     print('test pivot multiple values string')

#     df = gen_df_multiple_columns()

#     aggfunc_dict = {NAME_COL2: 'count', NAME_VALUE: 'median'}

#     # time

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     is_equal_pd = pivot_cython.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     assert is_equal_pd

def test_pivot_multiple_values_string_nunique():

    print()
    print('test pivot multiple values string nunique')

    df = gen_df_multiple_columns()

    aggfunc_dict = {NAME_COL2: 'nunique', NAME_VALUE: 'median'}

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=0, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd

def test_pivot_multiple_values():

    print()
    print('test pivot multiple_values')

    df = gen_df_multiple_values()

    # time

    aggfunc_dict = {NAME_VALUE: 'sum', NAME_VALUE2: 'min'}

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_VALUE, NAME_VALUE2], fill_value=0.0, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_VALUE, NAME_VALUE2], fill_value=0.0, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_multiple_values_fillNone():

    print()
    print('test pivot multiple values fillNone')

    df = gen_df_multiple_values()

    # time

    aggfunc_dict = {NAME_VALUE: 'median', NAME_VALUE2: 'sum'}

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_VALUE, NAME_VALUE2], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_VALUE, NAME_VALUE2], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_multiple_values_single_aggfunc():

    print()
    print('test pivot multiple_values format single aggfunc')

    df = gen_df_multiple_values()

    # time

    aggfunc_dict = {NAME_VALUE: 'max', NAME_VALUE2: 'mean'}

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_VALUE, NAME_VALUE2], fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_VALUE, NAME_VALUE2], fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_sum_int():

    print()
    print('test pivot sum int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd
    
def test_pivot_mean():

    print()
    print('test pivot mean')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_mean_fillNone():

    print()
    print('test pivot mean fill_value=None')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_mean_nodrop():

    print()
    print('test pivot mean fill_value=None, dropna=False')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='mean', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='mean', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_mean_int():
    # NOTE: pandas keeps mean as int if all entries in column are ints. 
    # this pivot_table always returns float.

    print()
    print('test pivot mean int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)
    # pivot_cython.info()

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='mean')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)
    # pivot_pandas.info()

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal

def test_pivot_std():

    print()
    print('test pivot std')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_cython_numpy = pivot_cython.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_cython_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_cython_numpy = np.nan_to_num(pivot_cython_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    # is_equal = (pivot_cython_numpy == pivot_pandas_numpy).all()
    # print('componentwise equal: ', is_equal)
    # is_equal_pd = pivot_cython.equals(pivot_pandas)
    # print('pd.equals: ', is_equal_pd)

    assert same_nan
    assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_pivot_std_int():

    print()
    print('test pivot std int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_cython_numpy = pivot_cython.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_cython_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_cython_numpy = np.nan_to_num(pivot_cython_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    # is_equal = (pivot_cython_numpy == pivot_pandas_numpy).all()
    # print('componentwise equal: ', is_equal)
    # is_equal_pd = pivot_cython.equals(pivot_pandas)
    # print('pd.equals: ', is_equal_pd)

    assert same_nan
    assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_pivot_max():

    print()
    print('test pivot max')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_max_nodrop():

    print()
    print('test pivot max no drop')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='max', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='max', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_max_nan_fill_none():

    print()
    print('test pivot max fill None')

    df = gen_df()
    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_max_nan_fill_nan():

    print()
    print('test pivot max fill nan')

    df = gen_df()
    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=np.nan, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=np.nan, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_max_int():

    print()
    print('test pivot max int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='max')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal

def test_pivot_min():

    print()
    print('test pivot min')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_min_nan_fill_none():

    print()
    print('test pivot min nan fill none')

    df = gen_df()
    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_min_nan_fill_nan():

    print()
    print('test pivot min nan fill nan')

    df = gen_df()
    df[NAME_VALUE][np.random.choice(a=[False, True], size=N_ROWS)] = np.nan

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=np.nan, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=np.nan, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_min_int():

    print()
    print('test pivot min int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='min')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal

def test_pivot_count():

    print()
    print('test pivot count')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='count')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd

def test_pivot_nunique():
    #TODO: better test (with actual nunique not equal to counts, and longer vectors per (i, j) pair)

    print()
    print('test pivot nunique')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd

def test_pivot_nunique_int():
    #TODO: better test (with actual nunique not equal to counts, and longer vectors per (i, j) pair)

    print()
    print('test pivot nunique int')

    df = gen_df_int()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0, aggfunc='nunique')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    #assert is_equal_pd

def test_pivot_median():

    print()
    print('test pivot median')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_pivot_median_fillNone():

    print()
    print('test pivot median fill None')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_sum_fill_none():

    print()
    print('test pivot sum with fill_value=None')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

# # def test_pivot_sum_fill_string():

# #     print('test pivot sum with fill_value="Nothing!"')

# #     df = gen_df()

# #     # time

# #     msg = 'cython'
# #     tick = time.perf_counter()
# #     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value='Nothing!', aggfunc='sum')
# #     print(msg, time.perf_counter() - tick)
# #     # print(pivot_cython)

# #     msg = 'pandas'
# #     tick = time.perf_counter()
# #     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value='Nothing!', aggfunc='sum')
# #     print(msg, time.perf_counter() - tick)
# #     # print(pivot_pandas)

# #     # check results are equal

# #     is_equal_pd = pivot_cython.equals(pivot_pandas)
# #     print('pd.equals: ', is_equal_pd)

# #     assert is_equal_pd

def test_pivot_std_fill():

    print()
    print('test pivot std fill_value=0.0')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_cython_numpy = pivot_cython.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_cython_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_cython_numpy = np.nan_to_num(pivot_cython_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    # is_equal = (pivot_cython_numpy == pivot_pandas_numpy).all()
    # print('componentwise equal: ', is_equal)
    # is_equal_pd = pivot_cython.equals(pivot_pandas)
    # print('pd.equals: ', is_equal_pd)

    assert same_nan
    assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_pivot_std_fillNone():

    print()
    print('test pivot std fill_value=None')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=None, aggfunc='std')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_cython_numpy = pivot_cython.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_cython_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_cython_numpy = np.nan_to_num(pivot_cython_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    # is_equal = (pivot_cython_numpy == pivot_pandas_numpy).all()
    # print('componentwise equal: ', is_equal)
    # is_equal_pd = pivot_cython.equals(pivot_pandas)
    # print('pd.equals: ', is_equal_pd)

    assert same_nan
    assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_pivot_std_fill_nodrop():

    print()
    print('test pivot std fill_value=0.0 dropna=False')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='std', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='std', dropna=False)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    pivot_cython_numpy = pivot_cython.to_numpy()
    pivot_pandas_numpy = pivot_pandas.to_numpy()
    same_nan = ((pivot_cython_numpy == np.nan) == (pivot_pandas_numpy == np.nan)).all()
    print('same NaN: ', same_nan)
    pivot_cython_numpy = np.nan_to_num(pivot_cython_numpy)
    pivot_pandas_numpy = np.nan_to_num(pivot_pandas_numpy)
    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython_numpy - pivot_pandas_numpy) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    # is_equal = (pivot_cython_numpy == pivot_pandas_numpy).all()
    # print('componentwise equal: ', is_equal)
    # is_equal_pd = pivot_cython.equals(pivot_pandas)
    # print('pd.equals: ', is_equal_pd)

    assert same_nan
    assert within_epsilon
    # assert is_equal
    # assert is_equal_pd

def test_multiple_columns():

    print()
    print('test pivot sum with multiple columns')

    df = gen_df_multiple_columns()

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_multiple_columns_nan():

    print()
    print('test pivot sum with multiple columns nan')

    df = gen_df_multiple_columns()
    df[NAME_COL][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan
    df[NAME_COL2][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_multiple_columns_median():

    print()
    print('test pivot median with multiple columns')

    df = gen_df_multiple_columns()

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL, NAME_COL2], values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_multiple_index():

    print()
    print('test pivot sum with multiple index')

    df = gen_df_multiple_index()

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_multiple_index_nan():

    print()
    print('test pivot sum with multiple index nan')

    df = gen_df_multiple_index()
    df[NAME_IDX][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan
    df[NAME_IDX2][np.random.choice(a=[False, True], size=N_ROWS, p=[0.75, 0.25])] = np.nan

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

def test_multiple_index_median():

    print()
    print('test pivot median with multiple index')

    df = gen_df_multiple_index()

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=[NAME_IDX, NAME_IDX2], columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='median')
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    epsilon = 1e-8
    within_epsilon = (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all()
    print('componentwise within {} :'.format(epsilon), within_epsilon)
    is_equal = (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all()
    print('componentwise equal: ', is_equal)
    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert within_epsilon
    assert is_equal
    assert is_equal_pd

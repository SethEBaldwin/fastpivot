import pivot
import pandas as pd
import numpy as np
import time

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

# Really good speed ups for these parameters
N_ROWS = 100000
N_COLS = 1000
N_IDX = 1000

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
    col3 = [x for x in np.random.randint(-10000, 10000, size=N_ROWS)]

    data = np.transpose([col1, col2, col3])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.int64)

    # print(df)

    return df

def gen_df_multiple_values():
    col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
    col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
    col3 = [x for x in np.random.normal(size=N_ROWS)]
    col4 = [x for x in np.random.randint(-10000, 10000, size=N_ROWS)]

    data = np.transpose([col1, col2, col3, col4])
    df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE, NAME_VALUE2], index=range(len(data)))
    df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)
    df[NAME_VALUE2] = df[NAME_VALUE2].astype(np.int64)

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

def test_pivot_nan_value():

    print()
    print('test pivot nan value')

    df = gen_df()

    df.iloc[0, -1] = np.nan

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

def test_pivot_nan_value_fillna0():

    print()
    print('test pivot nan value fillna=0')

    df = gen_df()

    df.iloc[0, -1] = np.nan

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

    df.iloc[0, 0] = np.nan

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

    df.iloc[0, 1] = np.nan

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

def test_pivot_multiple_values_list():

    print()
    print('test pivot multiple values list')

    df = gen_df_multiple_columns()

    aggfunc_dict = {NAME_COL2: 'nunique', NAME_VALUE: ['sum', 'median']}

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

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

def test_pivot_sum_silly():

    print()
    print('test pivot sum with index, columns list of single string')

    df = gen_df()

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=[NAME_IDX], columns=[NAME_COL], values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
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

def test_pivot_multiple_values_string():

    print()
    print('test pivot multiple values string')

    df = gen_df_multiple_columns()

    aggfunc_dict = {NAME_COL2: 'count', NAME_VALUE: 'median'}

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_multiple_values_string_nunique():

    print()
    print('test pivot multiple values string nunique')

    df = gen_df_multiple_columns()

    aggfunc_dict = {NAME_COL2: 'nunique', NAME_VALUE: 'median'}

    # time

    msg = 'cython'
    tick = time.perf_counter()
    pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_cython)

    msg = 'pandas'
    tick = time.perf_counter()
    pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=[NAME_COL2, NAME_VALUE], fill_value=None, aggfunc=aggfunc_dict)
    print(msg, time.perf_counter() - tick)
    # print(pivot_pandas)

    # check results are equal

    is_equal_pd = pivot_cython.equals(pivot_pandas)
    print('pd.equals: ', is_equal_pd)

    assert is_equal_pd

def test_pivot_multiple_values():

    print()
    print('test pivot multiple_values')

    df = gen_df_multiple_values()

    # time

    aggfunc_dict = {NAME_VALUE: 'mean', NAME_VALUE2: 'max'}

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

def test_pivot_multiple_values_single_aggfunc():

    print()
    print('test pivot multiple_values format single aggfunc')

    df = gen_df_multiple_values()

    # time

    aggfunc_dict = {NAME_VALUE: 'mean', NAME_VALUE2: 'max'}

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
    print('test pivot sum')

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
    assert is_equal_pd
    
def test_pivot_mean():

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

def test_pivot_mean_int():
    # NOTE: pandas keeps mean as int if all entries in column are ints. 
    # this pivot_table always returns float.

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
    # assert is_equal_pd

def test_pivot_std():

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

def test_pivot_max_int():

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
    assert is_equal_pd

def test_pivot_min():

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

def test_pivot_min_int():

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
    assert is_equal_pd

def test_pivot_count():

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
    assert is_equal_pd

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
    assert is_equal_pd

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
    assert is_equal_pd

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

def test_pivot_sum_fill_none():

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

# def test_pivot_sum_fill_string():

#     print('test pivot sum with fill_value="Nothing!"')

#     df = gen_df()

#     # time

#     msg = 'cython'
#     tick = time.perf_counter()
#     pivot_cython = pivot.pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value='Nothing!', aggfunc='sum')
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_cython)

#     msg = 'pandas'
#     tick = time.perf_counter()
#     pivot_pandas = df.pivot_table(index=NAME_IDX, columns=[NAME_COL], values=NAME_VALUE, fill_value='Nothing!', aggfunc='sum')
#     print(msg, time.perf_counter() - tick)
#     # print(pivot_pandas)

#     # check results are equal

#     is_equal_pd = pivot_cython.equals(pivot_pandas)
#     print('pd.equals: ', is_equal_pd)

#     assert is_equal_pd

def test_pivot_std_fill():

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

def test_pivot_std_fill_nodrop():

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

def test_multiple_columns_median():

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

def test_multiple_index_median():

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

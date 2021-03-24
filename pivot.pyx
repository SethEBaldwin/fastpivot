# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport sqrt
from libc.math cimport isnan
from cython.operator import dereference, postincrement
from libc.stdlib cimport malloc
from libcpp.algorithm cimport sort as stdsort
import pandas as pd
import numpy as np
cimport numpy as np
import time

# TODO: unit tests for types object (string), bool, date, timestamp, categorical
# TODO: make sure Datetime.date isn't converted to Timestamp
def pivot_table(df, index, columns, values, aggfunc='mean', fill_value=None, dropna=True):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    The main limitation is that you must aggregate, and you must do so by a list of preconstructed functions:
        For numerical values (np.float64 or np.int64), you can aggregate by any of 
           ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
        For other values, you can aggregate by any of 
            ['count', 'nunique']

    The arguments and return format mimic pandas very closely, with a few small differences:
    1) Occasionaly the ordering of the columns, such as when passing a list of aggfuncs and a single value column
    2) When passing values of type np.int64, values of type np.float64 will be returned for sum, mean, std, max, min,
        and median, and int.64 for count, and nunique. Pandas returns different types in some of these cases.
    3) edge cases are handled differently for the aggregation functions. The conventions are:
                            fastpivot       pandas
        sum of empty:       0.0             0.0 or 0
        mean of empty:      NaN             NaN
        std of empty:       NaN             NaN
        max of empty:       NaN             NaN
        min of empty:       NaN             NaN
        median of empty:    NaN             NaN
        count of empty:     0               NaN
        nunique of empty:   0               Nan
    4) The following arguments are not supported here: margins, margins_name, observed.

    Generally on a dataframe with many rows and many distinct values in the passed index and column, the performance of this
    pivot_tabel function beats pandas significantly, by a factor of 2 to 20.
    On a dataframe with many rows but few distinct values in the passed index and column, the speed of this pivot_table
    tends to be roughly on par with pandas.
    This pivot_table tends to be slower than pandas when there are many rows, multiple index or columns are passed,
    and the index and columns passed have few distinct values. 

    Arguments:
    df: pandas dataframe
    index: string or list, name(s) of column(s) that you want to become the index of the pivot table. 
    columns: string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values: string or list, name(s) of column(s) that contains as values the values of the pivot table.
    aggfunc: string, list, or dict, name of aggregation function. must be on implemented list above.
        if aggfunc is a dict, the format must be as in the following example:
        values = ['column_name1', 'column_name2', 'column_name3']
        aggfunc = {'column_name1': 'mean', 'column_name2': 'median', 'column_name3': ['count', 'nunique']}
    fill_value: scalar, value to replace missing values with in the pivot table.
    dropna: bool, if True rows and columns that are entirely NaN values will be dropped.

    Returns:
    pivot_df: pandas dataframe
    """
    
    #tick = time.perf_counter()
    assert isinstance(index, str) or isinstance(index, list)
    assert isinstance(columns, str) or isinstance(columns, list)
    if isinstance(index, str):
        df = df.dropna(subset=[index])
    if isinstance(index, list):
        df = df.dropna(subset=index)
        if len(index) == 1:
            index = index[0]
    if isinstance(columns, str):
        df = df.dropna(subset=[columns])
    if isinstance(columns, list):
        df = df.dropna(subset=columns)
        if len(columns) == 1:
            columns = columns[0]

    if isinstance(index, str):
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = df[index].factorize(sort=True)
        #print('factorize idx', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = pd.MultiIndex.from_frame(df[index]).factorize(sort=True)
        idx_arr_unique = pd.MultiIndex.from_tuples(idx_arr_unique, names=index)
        #print('factorize idx', time.perf_counter() - tick1)
    if isinstance(columns, str):
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = df[columns].factorize(sort=True)
        #print('tuple conversion col', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = pd.MultiIndex.from_frame(df[columns]).factorize(sort=True)
        col_arr_unique = pd.MultiIndex.from_tuples(col_arr_unique, names=columns)
        #print('factorize col', time.perf_counter() - tick1)
    #print(1, time.perf_counter() - tick)

    #tick = time.perf_counter()
    values_list, aggfunc_dict, keys = process_values_aggfunc(values, aggfunc)

    pivot_dfs = []
    for value in values_list:
        agg_list = aggfunc_dict[value]
        for agg in agg_list:
            pivot_df = pivot_compute_table(
                agg, 
                fill_value, 
                dropna, 
                index,
                columns,
                idx_arr, 
                col_arr, 
                df[value], 
                idx_arr_unique, 
                col_arr_unique
            )
            pivot_dfs.append(pivot_df)
    pivot_df = pd.concat(pivot_dfs, axis=1, keys=keys)
    #print(2, time.perf_counter() - tick)
    
    if dropna:
        pivot_df = pivot_df.dropna(axis=0, how='all')
        pivot_df = pivot_df.dropna(axis=1, how='all')
    if fill_value is not None:
        pivot_df = pivot_df.fillna(fill_value)

    return pivot_df

def process_values_aggfunc(values, aggfunc):
    # standardize format of values, aggfunc.
    # standardized means values is a sorted list of strings and
    # aggfunc is a dict with keys the strings in values and values
    # sorted lists of strings
    # keys gives the multicolumn keys for constructing pivot_df

    assert isinstance(values, str) or isinstance(values, list)
    
    if isinstance(values, str):
        values_list = [values]
    elif isinstance(values, list):
        values_list = sorted(values)

    assert isinstance(aggfunc, str) or isinstance(aggfunc, list) or isinstance(aggfunc, dict)
    
    if isinstance(aggfunc, str):
        aggfunc_dict = {value: [aggfunc] for value in values_list}
    elif isinstance(aggfunc, list):
        aggfunc_dict = {value: sorted(aggfunc) for value in values_list}
    elif isinstance(aggfunc, dict):
        def process(x):
            assert isinstance(x, str) or isinstance(x, list)
            if isinstance(x, str):
                return [x]
            elif isinstance(x, list):
                return sorted(x)
        aggfunc_dict = {value: process(aggfunc[value]) for value in values_list}
    
    # construct most general key
    keys = []
    for value in values_list:
        for agg in aggfunc_dict[value]:
            keys.append((value, agg))
    # if we don't need most general key, reduce layers of multicolumn
    if len(keys) == 1:
        keys = None
    elif len(values_list) == 1:
        keys = aggfunc_dict[values_list[0]]
    elif len(keys) == len(values_list):
        keys = values_list

    return values_list, aggfunc_dict, keys

def pivot_compute_table(aggfunc, fill_value, dropna, index, columns, idx_arr, col_arr, values_series, idx_arr_unique, col_arr_unique):

    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    pivot_arr = pivot_compute_agg(aggfunc, idx_arr, col_arr, values_series, n_idx, n_col)
    pivot_df = pd.DataFrame(pivot_arr, index=idx_arr_unique, columns=col_arr_unique)
    pivot_df.index.rename(index, inplace=True)
    pivot_df.columns.rename(columns, inplace=True)

    return pivot_df

def pivot_compute_agg(aggfunc, idx_arr, col_arr, values_series, n_idx, n_col):

    # handle types
    values_dtype = values_series.dtype
    if values_dtype == np.float64 or values_dtype == np.int64:
        numeric = True
        assert aggfunc in ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
        values_series = values_series.astype(np.float64)
    else:
        numeric = False
        assert aggfunc in ['count', 'nunique']
        #values_series = values_series.astype(str)
    
    # pivot and aggregate
    if numeric:
        if aggfunc == 'sum':
            pivot_arr = pivot_cython_sum(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col)
        elif aggfunc == 'mean':
            pivot_arr = pivot_cython_mean(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col)
        elif aggfunc == 'std':
            pivot_arr = pivot_cython_std(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col)
        elif aggfunc == 'max':
            pivot_arr = pivot_cython_max(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col)
        elif aggfunc == 'min':
            pivot_arr = pivot_cython_min(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col)
        elif aggfunc == 'count':
            nans_arr = values_series.isna().to_numpy()
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col, nans_arr)
        elif aggfunc == 'median':
            pivot_arr = pivot_cython_agg(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, median_cython)
        elif aggfunc == 'nunique':
            pivot_arr = pivot_cython_agg_int(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, nunique_cython)
    else:
        if aggfunc == 'count':
            nans_arr = values_series.isna().to_numpy()
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col, nans_arr)
        elif aggfunc == 'nunique':
            values_arr, _ = values_series.factorize()
            values_arr = values_arr.astype(np.float64) # TODO: unit tests... careful with nans?
            pivot_arr = pivot_cython_agg_int(idx_arr, col_arr, values_arr, n_idx, n_col, nunique_cython)
    arr = np.array(pivot_arr)

    return arr

cdef double[:, :] pivot_cython_sum(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    nans = np.zeros((N, M), dtype=np.float64)
    nans.fill(np.nan)
    cdef double[:, :] pivot_arr = nans
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        #####################################
        # pandas considers empty sums to be 0
        if isnan(value):
            if isnan(pivot_arr[i, j]):
                pivot_arr[i, j] = 0.0
        #####################################
        else:
            if not isnan(pivot_arr[i, j]):
                pivot_arr[i, j] += value
            else:
                pivot_arr[i, j] = value
    return pivot_arr

cdef double[:, :] pivot_cython_mean(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    nans = np.zeros((N, M), dtype=np.float64)
    nans.fill(np.nan)
    cdef double[:, :] pivot_arr = nans
    cdef double[:, :] pivot_counts_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    cdef double divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            if not isnan(pivot_arr[i, j]):
                pivot_arr[i, j] += value
            else:
                pivot_arr[i, j] = value
            pivot_counts_arr[i, j] += 1.0
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:
                pivot_arr[i, j] /= divisor
    return pivot_arr

cdef double[:, :] pivot_cython_std(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    nans1 = np.zeros((N, M), dtype=np.float64)
    nans1.fill(np.nan)
    cdef double[:, :] pivot_arr = nans1
    nans2 = np.zeros((N, M), dtype=np.float64)
    nans2.fill(np.nan)
    cdef double[:, :] mean_arr = nans2
    cdef double[:, :] pivot_counts_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    cdef double divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            if not isnan(mean_arr[i, j]):
                mean_arr[i, j] += value
            else:
                mean_arr[i, j] = value
            pivot_counts_arr[i, j] += 1.0
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:
                mean_arr[i, j] /= divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        if not isnan(value_arr[k]):
            if not isnan(pivot_arr[i, j]):
                value = value_arr[k] - mean_arr[i, j]
                pivot_arr[i, j] += (value*value)
            else:
                value = value_arr[k] - mean_arr[i, j]
                pivot_arr[i, j] = (value*value)
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:# and divisor != 1.0:
                pivot_arr[i, j] /= (divisor - 1)
                pivot_arr[i, j] = sqrt(pivot_arr[i, j])
    return pivot_arr

cdef double[:, :] pivot_cython_max(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    nans = np.zeros((N, M), dtype=np.float64)
    nans.fill(np.nan)
    cdef double[:, :] pivot_arr = nans
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            if not isnan(pivot_arr[i, j]):
                if pivot_arr[i, j] < value:
                    pivot_arr[i, j] = value
            else:
                pivot_arr[i, j] = value
    return pivot_arr

cdef double[:, :] pivot_cython_min(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    nans = np.zeros((N, M), dtype=np.float64)
    nans.fill(np.nan)
    cdef double[:, :] pivot_arr = nans
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            if not isnan(pivot_arr[i, j]):
                if pivot_arr[i, j] > value:
                    pivot_arr[i, j] = value
            else:
                pivot_arr[i, j] = value
    return pivot_arr

cdef long[:, :] pivot_cython_count(long[:] idx_arr, long[:] col_arr, int N, int M, bool[:] nans_arr):
    cdef long[:, :] pivot_arr = np.zeros((N, M), dtype=np.int64)
    cdef int i, j, k
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        if not nans_arr[k]:
            pivot_arr[i, j] += 1
    return pivot_arr

cdef bool[:, :] find_missing_cython(long[:] idx_arr, long[:] col_arr, int N, int M):
    cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        missing_arr[i, j] = 0
    return missing_arr

cdef bool[:, :] find_missing_cython_nan(long[:] idx_arr, long[:] col_arr, int N, int M, bool[:] nans_arr):
    cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        if not nans_arr[k]:
            missing_arr[i, j] = 0
    return missing_arr

cdef bool[:, :] find_missing_std_cython(long[:] idx_arr, long[:] col_arr, int N, int M):
    cdef bool[:, :] exists_arr = np.zeros((N, M), dtype=np.bool)
    cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        if exists_arr[i, j]:
            missing_arr[i, j] = 0
        else:
            exists_arr[i, j] = 1
    return missing_arr

ctypedef double vec_to_double(vector[double] &)
ctypedef long vec_to_long(vector[double] &)

cdef double sum_cython(vector[double] &vec):
    cdef int k
    cdef double value = 0.0
    for k in range(vec.size()):
        value += vec[k]
    return value

cdef double median_cython(vector[double] &vec):
    cdef int k = 0
    cdef int mid = vec.size() / 2
    cdef double med
    cdef int i = 0
    cdef int j = vec.size() - 1
    cdef double max_left

    if vec.size() == 0:
        return np.nan
    
    while k != mid:
        k = partition(vec, mid, i, j)
        if k < mid:
            i = k
        elif k > mid:
            j = k
    if vec.size() % 2 == 1:
        med = vec[mid]
    else:
        max_left = max_left_cython(vec, mid - 1)
        med = (vec[mid] + max_left) / 2

    return med

cdef int partition(vector[double] &vec, int mid, int i, int j):
    cdef double x = vec[mid]
    cdef double temp
    while True:
        while vec[i] < x:
            i += 1
        while vec[j] > x:
            j -= 1
        if i == j:
            return j
        else:
            temp = vec[i]
            vec[i] = vec[j]
            vec[j] = temp

cdef double max_left_cython(vector[double] & vec, int upper_idx):
    # never called on empty vec
    cdef int i
    cdef double value
    value = vec[0]
    for i in range(1, upper_idx + 1):
        if value < vec[i]:
            value = vec[i]
    return value

cdef long nunique_cython(vector[double] &vec):
    cdef int k
    cdef cset[double] value_set
    cdef long n
    for k in range(vec.size()):
        value_set.insert(vec[k])
    n = value_set.size()    
    return n

cdef double[:, :] pivot_cython_agg(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_double agg):
    #tick = time.perf_counter()
    cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
    cdef vector[double].iterator it
    cdef double[:, :] pivot_arr_return = np.zeros((N, M), dtype=np.float64)
    cdef int i, j
    cdef double value
    #print('cdef and initialize', time.perf_counter() - tick)
    #tick = time.perf_counter()
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            pivot_arr[i*M + j].push_back(value)
    #print('push_back', time.perf_counter() - tick)
    #tick = time.perf_counter()
    for i in range(N):
        for j in range(M):
                value = agg(pivot_arr[i*M + j])
                pivot_arr_return[i, j] = value
    #print('agg and assign', time.perf_counter() - tick)
    return pivot_arr_return

cdef long[:, :] pivot_cython_agg_int(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_long agg):
    #tick = time.perf_counter()
    cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
    cdef vector[double].iterator it
    cdef long[:, :] pivot_arr_return = np.zeros((N, M), dtype=np.int64)
    cdef int i, j
    cdef double value
    cdef long value_int
    #print('cdef and initialize', time.perf_counter() - tick)
    #tick = time.perf_counter()
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if not isnan(value):
            pivot_arr[i*M + j].push_back(value)
    #print('push_back', time.perf_counter() - tick)
    #tick = time.perf_counter()
    for i in range(N):
        for j in range(M):
                value_int = agg(pivot_arr[i*M + j])
                pivot_arr_return[i, j] = value_int
    #print('agg and assign', time.perf_counter() - tick)
    return pivot_arr_return

# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport sqrt
from cython.operator import dereference, postincrement
from libc.stdlib cimport malloc
from libcpp.algorithm cimport sort as stdsort
import pandas as pd
import numpy as np
cimport numpy as np
import time

# TODO: convenience like fill_value dict? aggfunc list or dict with values lists?
# TODO: handle other dtypes for nunique
# TODO: std is slow because of processing at the end
# TODO: faster median
# TODO: address type conversions: have example where column with type Datetime.date was converted to Timestamp
def pivot_table(df, index, columns, values, aggfunc='mean', fill_value=None, dropna=True):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Values must not contain any NaNs.
    For numerical values (np.float64 or np.int64), aggregates by any of 
        ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
    For other values, aggregates by any of 
        ['count', 'nunique']
    
    Arguments:
    df: pandas dataframe
    index: string or list, name(s) of column(s) that you want to become the index of the pivot table. 
    columns: string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values: string or list, name(s) of column(s) that contains as values the values of the pivot table.
    aggfunc: string or dict, name of aggregation function. must be on implemented list above.
        if aggfunc is a dict, the format must be as in the following example:
        values = ['column_name1', 'column_name2', 'column_name3']
        aggfunc = {'column_name1': 'mean', 'column_name2': 'median', 'column_name3': 'nunique'}
        we currently don't support lists of aggfuncs, and when aggfunct is a dict, we do not support its values being lists 
        of aggfuncs - they must be strings
    fill_value: scalar, value to replace missing values with in the pivot table.
    dropna: bool, if True rows and columns that are entirely NaN values will be dropped.

    Returns:
    pivot_df: pandas dataframe
    """
    
    tick = time.perf_counter()
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
    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    print(1, time.perf_counter() - tick)

    tick = time.perf_counter()
    if isinstance(values, str):
        pivot_arr = pivot_compute_agg(aggfunc, idx_arr, col_arr, df[values], n_idx, n_col)
        pivot_df = pd.DataFrame(pivot_arr, index=idx_arr_unique, columns=col_arr_unique)
        pivot_df = pivot_fill(aggfunc, fill_value, dropna, pivot_df, idx_arr, col_arr, n_idx, n_col)
        pivot_df.index.rename(index, inplace=True)
        pivot_df.columns.rename(columns, inplace=True)
    else:  
        if isinstance(aggfunc, str):
            aggfunc_dict = {value: aggfunc for value in values}
        # elif isinstance(aggfunc, list):
        #     aggfunc_dict = {value: agg for value, agg in zip(values, aggfunc)}
        else:
            aggfunc_dict = aggfunc
        pivot_dfs = []
        for value in values:
            pivot_arr = pivot_compute_agg(aggfunc_dict[value], idx_arr, col_arr, df[value], n_idx, n_col)
            pivot_df = pd.DataFrame(pivot_arr, index=idx_arr_unique, columns=col_arr_unique)
            pivot_df = pivot_fill(aggfunc_dict[value], fill_value, dropna, pivot_df, idx_arr, col_arr, n_idx, n_col)
            pivot_df.index.rename(index, inplace=True)
            pivot_df.columns.rename(columns, inplace=True)
            pivot_dfs.append(pivot_df)
        # if isinstance(aggfunc, list):
        #     keys = zip(aggfunc, values)
        # else: 
            # keys = values
        pivot_df = pd.concat(pivot_dfs, axis=1, keys=values)
    print(2, time.perf_counter() - tick)
    
    return pivot_df

def pivot_compute_agg(aggfunc, idx_arr, col_arr, values_series, n_idx, n_col):

    # handle types
    values_dtype = values_series.dtype
    assert not values_series.isnull().to_numpy().any()
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
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col)
        elif aggfunc == 'median':
            pivot_arr = pivot_cython_agg(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, median_cython)
        elif aggfunc == 'nunique':
            pivot_arr = pivot_cython_agg_int(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, nunique_cython)
    else:
        if aggfunc == 'count':
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col)
        elif aggfunc == 'nunique':
            values_arr, _ = values_series.factorize()
            values_arr = values_arr.astype(np.float64)
            pivot_arr = pivot_cython_agg_int(idx_arr, col_arr, values_arr, n_idx, n_col, nunique_cython)
    arr = np.array(pivot_arr)

    # handle type conversion back if sensible
    if values_dtype == np.int64:
        if aggfunc in ['sum', 'max', 'min']:
            arr = arr.astype(np.int64)
    return arr

def pivot_fill(aggfunc, fill_value, dropna, pivot_df, idx_arr, col_arr, n_idx, n_col):
    if aggfunc == 'std' and (dropna or fill_value != 0):
        missing_arr_cython = find_missing_std_cython(idx_arr, col_arr, n_idx, n_col)
        missing_arr = np.array(missing_arr_cython)
        pivot_df[missing_arr] = np.nan
        if dropna:
            pivot_df = pivot_df.dropna(axis=0, how='all')
            pivot_df = pivot_df.dropna(axis=1, how='all')
        if fill_value is not None:
            pivot_df = pivot_df.fillna(value=fill_value)
    elif fill_value != 0:
        missing_arr_cython = find_missing_cython(idx_arr, col_arr, n_idx, n_col)
        missing_arr = np.array(missing_arr_cython)
        pivot_df[missing_arr] = fill_value
    return pivot_df

cdef double[:, :] pivot_cython_sum(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        pivot_arr[i, j] += value
    return pivot_arr

cdef double[:, :] pivot_cython_mean(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef double[:, :] pivot_counts_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    cdef double divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        pivot_arr[i, j] += value
        pivot_counts_arr[i, j] += 1.0
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:
                pivot_arr[i, j] /= divisor
    return pivot_arr

cdef double[:, :] pivot_cython_std(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef double[:, :] mean_arr = np.zeros((N, M), dtype=np.float64)
    cdef double[:, :] pivot_counts_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    cdef double divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        mean_arr[i, j] += value
        pivot_counts_arr[i, j] += 1.0
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:
                mean_arr[i, j] /= divisor
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k] - mean_arr[i, j]
        pivot_arr[i, j] += (value*value)
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0 and divisor != 1.0:
                pivot_arr[i, j] /= (divisor - 1)
                pivot_arr[i, j] = sqrt(pivot_arr[i, j])
    return pivot_arr

cdef double[:, :] pivot_cython_max(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef bool[:, :] exists_arr = np.zeros((N, M), dtype=np.bool)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if exists_arr[i, j]:
            if pivot_arr[i, j] < value:
                pivot_arr[i, j] = value
        else:
            pivot_arr[i, j] = value
            exists_arr[i, j] = 1
    return pivot_arr

cdef double[:, :] pivot_cython_min(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef bool[:, :] exists_arr = np.zeros((N, M), dtype=np.bool)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        if exists_arr[i, j]:
            if pivot_arr[i, j] > value:
                pivot_arr[i, j] = value
        else:
            pivot_arr[i, j] = value
            exists_arr[i, j] = 1
    return pivot_arr

cdef long[:, :] pivot_cython_count(long[:] idx_arr, long[:] col_arr, int N, int M):
    cdef long[:, :] pivot_arr = np.zeros((N, M), dtype=np.int64)
    cdef int i, j, k
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
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

# TODO faster algorithm?
cdef double median_cython(vector[double] &vec):
    cdef int k
    cdef int idx
    cdef double value = 0.0
    cdef double med
    stdsort(vec.begin(), vec.end())
    idx = vec.size() / 2
    if vec.size() % 2 == 1:
        med = vec[idx]
    elif vec.size() == 0:
        med = 0.0
    else:
        med = (vec[idx] + vec[idx-1]) / 2
    return med

cdef long nunique_cython(vector[double] &vec):
    cdef int k
    cdef cset[double] value_set
    cdef long n
    for k in range(vec.size()):
        value_set.insert(vec[k])
    n = value_set.size()    
    return n

cdef double[:, :] pivot_cython_agg(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_double agg):
    tick = time.perf_counter()
    cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
    cdef vector[double].iterator it
    cdef double[:, :] pivot_arr_return = np.zeros((N, M), dtype=np.float64)
    cdef int i, j
    cdef double value
    print('cdef and initialize', time.perf_counter() - tick)
    tick = time.perf_counter()
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        pivot_arr[i*M + j].push_back(value)
    print('push_back', time.perf_counter() - tick)
    tick = time.perf_counter()
    for i in range(N):
        for j in range(M):
                value = agg(pivot_arr[i*M + j])
                pivot_arr_return[i, j] = value
    print('agg and assign', time.perf_counter() - tick)
    return pivot_arr_return

cdef long[:, :] pivot_cython_agg_int(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_long agg):
    tick = time.perf_counter()
    cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
    cdef vector[double].iterator it
    cdef long[:, :] pivot_arr_return = np.zeros((N, M), dtype=np.int64)
    cdef int i, j
    cdef double value
    cdef long value_int
    print('cdef and initialize', time.perf_counter() - tick)
    tick = time.perf_counter()
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        pivot_arr[i*M + j].push_back(value)
    print('push_back', time.perf_counter() - tick)
    tick = time.perf_counter()
    for i in range(N):
        for j in range(M):
                value_int = agg(pivot_arr[i*M + j])
                pivot_arr_return[i, j] = value_int
    print('agg and assign', time.perf_counter() - tick)
    return pivot_arr_return

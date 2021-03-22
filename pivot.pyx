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

# TODO: address type conversions: have example where column with type Datetime.date was converted to Timestamp
def pivot_table(df, index, columns, values, aggfunc='mean', fill_value=None):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Aggregates by any of ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
    Values must be of type np.float64 and must not contain any NaNs.
    Arguments:
    df: pandas dataframe
    index: string or list, name(s) of column(s) that you want to become the index of the pivot table. 
    columns: string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values: string, name of column that contains as values the values of the pivot table. values must be of type np.float64.
        values must not contain NaNs.
    aggfunc: string, name of aggregation function. must be on implemented list above.
    fill_value: scalar, value to replace missing values with in the pivot table.
    Returns a pandas dataframe
    """
    assert aggfunc in ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
    tick = time.perf_counter()
    if isinstance(index, str):
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = df[index].factorize(sort=True)
        #print('factorize idx', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = pd.MultiIndex.from_frame(df[index]).factorize(sort=True)
        #idx_arr, idx_arr_unique = df.set_index(index).index.factorize(sort=True)
        #print('factorize idx', time.perf_counter() - tick1)
    if isinstance(columns, str):
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = df[columns].factorize(sort=True)
        #print('tuple conversion col', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = pd.MultiIndex.from_frame(df[columns]).factorize(sort=True)
        #col_arr, col_arr_unique = df.set_index(columns).index.factorize(sort=True)
        #print('factorize col', time.perf_counter() - tick1)
    print(1, time.perf_counter() - tick)
    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    tick = time.perf_counter()
    if aggfunc == 'sum':
        pivot_arr = pivot_cython_sum(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif aggfunc == 'mean':
        pivot_arr = pivot_cython_mean(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif aggfunc == 'std':
        pivot_arr = pivot_cython_std(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif aggfunc == 'max':
        pivot_arr = pivot_cython_max(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif aggfunc == 'min':
        pivot_arr = pivot_cython_min(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif aggfunc == 'count':
        pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col)
    elif aggfunc == 'median':
        pivot_arr = pivot_cython_agg(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col, median_cython)
    elif aggfunc == 'nunique':
        pivot_arr = pivot_cython_agg_int(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col, nunique_cython)
    print(2, time.perf_counter() - tick)
    #tick = time.perf_counter()
    arr = np.array(pivot_arr)
    if not isinstance(index, str):
        idx_arr_unique = pd.MultiIndex.from_tuples(idx_arr_unique, names=index)
    if not isinstance(columns, str):
        col_arr_unique = pd.MultiIndex.from_tuples(col_arr_unique, names=columns)
    pivot_df = pd.DataFrame(arr, index=idx_arr_unique, columns=col_arr_unique)
    pivot_df.index.rename(index, inplace=True)
    pivot_df.columns.rename(columns, inplace=True)
    if aggfunc == 'std':
        missing_arr_cython = find_missing_std_cython(idx_arr, col_arr, n_idx, n_col)
        missing_arr = np.array(missing_arr_cython)
        pivot_df[missing_arr] = np.nan
        # pandas pivot_table automatically removes rows and cols that are all NaN
        pivot_df = pivot_df.dropna(axis=0, how='all')
        pivot_df = pivot_df.dropna(axis=1, how='all')
        if fill_value is not None:
            pivot_df = pivot_df.fillna(value=fill_value)
    elif fill_value != 0:
        missing_arr_cython = find_missing_cython(idx_arr, col_arr, n_idx, n_col)
        missing_arr = np.array(missing_arr_cython)
        pivot_df[missing_arr] = fill_value
    #print(3, time.perf_counter() - tick)
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

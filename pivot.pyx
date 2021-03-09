# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from cython.operator import dereference, postincrement
import pandas as pd
import numpy as np
cimport numpy as np
import time

# TODO: address type conversions: have example where column with type Datetime.date was converted to Timestamp
def pivot(df, index, columns, values, agg='mean'):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Fills by 0.0, currently aggregates by either sum or mean.
    Arguments:
    df: pandas dataframe
    index: string, name of column that you want to become the index. 
    columns: string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values: string, name of column that contains as values the values of the pivot table. values must be of type np.float64.
    agg: string, name of aggregation function. must be 'sum' or 'mean'.
    Returns a pandas dataframe
    """
    assert agg in ['sum', 'mean']
    tick = time.perf_counter()
    if isinstance(index, str):
        tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = df[index].factorize(sort=True)
        print('factorize idx', time.perf_counter() - tick1)
    else:
        tick1 = time.perf_counter()
        index_series = pd.Series([tuple(x) for x in df[index].to_numpy()])
        print('tuple conversion idx', time.perf_counter() - tick1)
        tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = index_series.factorize(sort=True)
        print('factorize idx', time.perf_counter() - tick1)
    if isinstance(columns, str):
        tick1 = time.perf_counter()
        col_arr, col_arr_unique = df[columns].factorize(sort=True)
        print('tuple conversion col', time.perf_counter() - tick1)
    else:
        tick1 = time.perf_counter()
        columns_series = pd.Series([tuple(x) for x in df[columns].to_numpy()])
        print('tuple conversion col', time.perf_counter() - tick1)
        tick1 = time.perf_counter()
        col_arr, col_arr_unique = columns_series.factorize(sort=True)
        print('factorize col', time.perf_counter() - tick)
    print(1, time.perf_counter() - tick)
    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    tick = time.perf_counter()
    if agg == 'sum':
        pivot_arr = pivot_cython_sum(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif agg == 'mean':
        pivot_arr = pivot_cython_mean(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
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

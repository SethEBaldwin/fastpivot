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

# TODO: allow by to be list
def groupby(df, by, agg='mean'):
    """
    A very basic and limited, but hopefully fast implementation of groupby. (CURRENTLY NOT FAST!)
    Currently aggregates by either sum or mean.
    Arguments:
    df: pandas dataframe. All values must be np.float64 except for those in the column for by
    by: string, name of column which will become index.
    agg: string, name of aggregation function. must be 'sum' or 'mean'.
    Returns a pandas dataframe.
    """
    assert agg in ['sum', 'mean']
    tick = time.perf_counter()
    # TODO: this is too slow here, need to replace with faster function
    idx_arr, idx_arr_unique = df[by].factorize(sort=True)
    n_idx = idx_arr_unique.shape[0]
    remaining_columns = [x for x in df.columns if x != by]
    n_col = len(remaining_columns)
    print(1, time.perf_counter() - tick)
    tick = time.perf_counter()
    if agg == 'sum':
        groupby_arr = groupby_cython_sum(idx_arr, df[remaining_columns].to_numpy(), n_idx, n_col)
    elif agg == 'mean':
        groupby_arr = groupby_cython_mean(idx_arr, df[remaining_columns].to_numpy(), n_idx, n_col)
    print(2, time.perf_counter() - tick)
    #tick = time.perf_counter()
    arr = np.array(groupby_arr)
    groupby_df = pd.DataFrame(arr, index=idx_arr_unique, columns=remaining_columns)
    groupby_df.index.rename(by, inplace=True)
    #print(3, time.perf_counter() - tick)
    return groupby_df

cdef double[:, :] groupby_cython_sum(long[:] idx_arr, double[:, :] value_arr, int N, int M):
    cdef double[:, :] groupby_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        for j in range(M):
            i = idx_arr[k]
            value = value_arr[k, j]
            groupby_arr[i, j] += value
    return groupby_arr

cdef double[:, :] groupby_cython_mean(long[:] idx_arr, double[:, :] value_arr, int N, int M):
    cdef double[:, :] groupby_arr = np.zeros((N, M), dtype=np.float64)
    cdef double[:, :] groupby_counts_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    cdef double divisor
    for k in range(idx_arr.shape[0]):
        for j in range(M):
            i = idx_arr[k]
            value = value_arr[k, j]
            groupby_arr[i, j] += value
            groupby_counts_arr[i, j] += 1.0
    for i in range(N):
        for j in range(M):
            divisor = groupby_counts_arr[i, j]
            if divisor != 0.0:
                groupby_arr[i, j] /= divisor
    return groupby_arr

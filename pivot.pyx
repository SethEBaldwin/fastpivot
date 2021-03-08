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

# ctypedef double (*f_vec_to_double)(vector[double])

# cdef cmap[string, int] dict_to_cmap(dict the_dict):
#     # the_dict is a dictionary mapping strings to ints
#     cdef string map_key
#     cdef int map_val
#     cdef cpair[string, int] map_element
#     cdef cmap[string, int] my_map
#     for key,val in the_dict.items():
#         map_key = key
#         map_val = val
#         map_element = (map_key, map_val)
#         my_map.insert(map_element)
#     return my_map

# def array_to_vector(arr):
#     # arr is a 1d numpy array of strings
#     cdef string element
#     cdef int i
#     cdef vector[string] my_vector
#     for i in range(arr.shape[0]):
#         element = arr[i]
#         my_vector.push_back(element)
#     return my_vector

# def enumerate(arr):
#     # takes 1d np array of strings and returns an array of ints, where 0 corresponds to first unique element of arr, 1 the second, etc.
#     cdef cmap[string, int] enumeration_map
#     cdef cpair[string, int] map_pair
#     cdef string value
#     cdef int is_old
#     cdef long count = 0
#     cdef int n = arr.shape[0]
#     cdef long[:] return_arr = np.empty(n, dtype=np.int)
#     cdef int i
#     for i in range(n):
#         value = arr[i]
#         is_old = enumeration_map.count(value)
#         if is_old:
#             return_arr[i] = enumeration_map[value]
#         else:
#             return_arr[i] = count
#             map_pair = (value, count)
#             enumeration_map.insert(map_pair)
#             count += 1            
#     return return_arr

# def enumerate2(arr, unique_arr):
#     # takes 1d np array of strings 1d array of unique strings from arr.
#     # returns an array of ints, where 0 corresponds to first unique element of arr, 1 the second, etc.
#     cdef cmap[string, int] enumeration_map
#     cdef cpair[string, int] map_pair
#     cdef string value
#     cdef int n = arr.shape[0]
#     cdef int m = unique_arr.shape[0]
#     cdef long[:] return_arr = np.empty(n, dtype=np.int)
#     cdef int i, j
#     for j in range(m):
#         value = unique_arr[j]
#         map_pair = (value, j)
#         enumeration_map.insert(map_pair)
#     for i in range(n):
#         value = arr[i]
#         return_arr[i] = enumeration_map[value]
#     return return_arr

# TODO: general multi index / multi columns
def pivot(df, index, columns, values, agg='mean'):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Fills by 0.0, currently aggregates by either sum or mean.
    Arguments:
    df: pandas dataframe
    index: string, name of column that you want to become the index. 
    columns: string or list, name(s) of column that contains as values the columns of the pivot table. 
    values: string, name of column that contains as values the values of the pivot table. values must be of type np.float64.
    agg: string, name of aggregation function. must be 'sum' or 'mean'.
    Returns a pandas dataframe
    """
    assert agg in ['sum', 'mean']
    #tick = time.perf_counter()
    idx_arr, idx_arr_unique = df[index].factorize(sort=True)
    if isinstance(columns, str):
        col_arr, col_arr_unique = df[columns].factorize(sort=True)
    #elif len(columns) == 2 and isinstance(df.loc[0, columns[0]], str) and isinstance(df.loc[0, columns[1]], str):
    #    tick = time.perf_counter()
    #    columns_pairs_series = pd.Series(to_pairs(df[columns].to_numpy()))
    #    print('to pairs', time.perf_counter() - tick)
    #    col_arr, col_arr_unique = columns_pairs_series.factorize(sort=True)
    else:
        tick1 = time.perf_counter()
        #columns_series = df[columns].apply(lambda x: tuple(x), axis=1)
        columns_series = pd.Series([tuple(x) for x in df[columns].to_numpy()])
        print('apply', time.perf_counter() - tick1)
        col_arr, col_arr_unique = columns_series.factorize(sort=True)
        #return df.pivot_table(index=index, columns=columns, values=values, fill_value=0.0, aggfunc=agg)
    #print(1, time.perf_counter() - tick)
    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    #tick = time.perf_counter()
    if agg == 'sum':
        pivot_arr = pivot_cython_sum(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    elif agg == 'mean':
        pivot_arr = pivot_cython_mean(idx_arr, col_arr, df[values].to_numpy(), n_idx, n_col)
    #print(2, time.perf_counter() - tick)
    #tick = time.perf_counter()
    arr = np.array(pivot_arr)
    if isinstance(columns, str):
        pivot_df = pd.DataFrame(arr, index=idx_arr_unique, columns=col_arr_unique)
        pivot_df.columns.rename(columns, inplace=True)
    else:
        col_arr_unique_multi = pd.MultiIndex.from_tuples(col_arr_unique, names=columns)
        pivot_df = pd.DataFrame(arr, index=idx_arr_unique, columns=col_arr_unique_multi)
    pivot_df.index.rename(index, inplace=True)
    #print(3, time.perf_counter() - tick)
    return pivot_df

# cdef vector[cpair[string, string]] to_pairs(np.ndarray arr):
#     cdef cpair[string, string] tup
#     cdef int i
#     cdef vector[cpair[string, string]] pairs
#     for i in range(arr.shape[0]):
#         tup = (arr[i, 0], arr[i, 1])
#         pairs.push_back(tup)
#     return pairs

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
        # print(i, j, value)
    for i in range(N):
        for j in range(M):
            divisor = pivot_counts_arr[i, j]
            if divisor != 0.0:
                pivot_arr[i, j] /= divisor
            # print(i, j, divisor)
    return pivot_arr

# cdef double[:, :] pivot_cython_agg(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, f_vec_to_double aggfunc):
#     cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
#     cdef cpair[int, int] coords
#     cdef cmap[cpair[int, int], vector[double]] pivot_map
#     cdef int i, j, k
#     cdef double value
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         value = value_arr[k]
#         coords = (i, j)
#         if pivot_map.count(coords):
#             pivot_map[coords].push_back(value)
#         else:
#             pivot_map[coords] = [value]

#     cdef cmap[cpair[int, int], vector[double]].iterator it = pivot_map.begin()
#     while it != pivot_map.end():
#         coords = dereference(it).first # key
#         i = coords.first
#         j = coords.second
#         values = dereference(it).second # value
#         pivot_arr[i, j] = aggfunc(values)
#         postincrement(it)
#     return pivot_arr

# cdef double vec_sum(vector[double] vec):
#     cdef double result = 0
#     cdef int i
#     for i in range(vec.size()):
#         result += vec[i]
#     return result

# cdef double vec_mean(vector[double] vec):
#     cdef double result = 0
#     cdef double divisor
#     cdef int i
#     for i in range(vec.size()):
#         result += vec[i]
#     if vec.size() != 0:
#         divisor = vec.size()
#         result = result / divisor
#     return result
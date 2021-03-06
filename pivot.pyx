# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from cython.operator import dereference, postincrement
import pandas as pd
import numpy as np
import time

ctypedef double (*f_vec_to_double)(vector[double])

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

def pivot(df, index, column, value, agg='sum'):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Aggregates by sum, fills by 0.0
    Arguments:
    df: pandas dataframe
    index: string, name of column that you want to become the index. 
    column: string, name of column that contains as values the columns of the pivot table. 
    value: string, name of column that contains as values the values of the pivot table. values must be of type np.float64.
    Returns a pandas dataframe
    """
    #tick = time.perf_counter()
    idx_arr, idx_arr_unique = df[index].factorize(sort=True)
    col_arr, col_arr_unique = df[column].factorize(sort=True)
    #print(1, time.perf_counter() - tick)
    n_idx = idx_arr_unique.shape[0]
    n_col = col_arr_unique.shape[0]
    #tick = time.perf_counter()
    if agg == 'orig':
        pivot_arr = pivot_cython(idx_arr, col_arr, df[value].to_numpy(), n_idx, n_col)
    else:
        if agg == 'sum':
            aggfunc = vec_sum
        if agg == 'mean':
            aggfunc = vec_mean
        pivot_arr = pivot_cython_agg(idx_arr, col_arr, df[value].to_numpy(), n_idx, n_col, aggfunc)
    #print(2, time.perf_counter() - tick)
    #tick = time.perf_counter()
    arr = np.array(pivot_arr)
    pivot_df = pd.DataFrame(arr, index=idx_arr_unique, columns=col_arr_unique)
    #print(3, time.perf_counter() - tick)
    return pivot_df

cdef double[:, :] pivot_cython(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        pivot_arr[i, j] += value
    return pivot_arr

cdef double[:, :] pivot_cython_agg(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, f_vec_to_double aggfunc):
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef cpair[int, int] coords
    cdef cmap[cpair[int, int], vector[double]] pivot_map
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        coords = (i, j)
        if pivot_map.count(coords):
            pivot_map[coords].push_back(value)
        else:
            pivot_map[coords] = [value]

    cdef cmap[cpair[int, int], vector[double]].iterator it = pivot_map.begin()
    while it != pivot_map.end():
        coords = dereference(it).first # key
        i = coords.first
        j = coords.second
        values = dereference(it).second # value
        pivot_arr[i, j] = aggfunc(values)
        postincrement(it)
    return pivot_arr

cdef double vec_sum(vector[double] vec):
    cdef double result = 0
    cdef int i
    for i in range(vec.size()):
        result += vec[i]
    return result

cdef double vec_mean(vector[double] vec):
    cdef double result = 0
    cdef double divisor
    cdef int i
    for i in range(vec.size()):
        result += vec[i]
    if vec.size() != 0:
        divisor = vec.size()
        result = result / divisor
    return result
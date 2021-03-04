# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
import pandas as pd
import numpy as np

cdef cmap[string, int] dict_to_cmap(dict the_dict):
    # the_dict is a dictionary mapping strings to ints
    cdef string map_key
    cdef int map_val
    cdef cpair[string, int] map_element
    cdef cmap[string, int] my_map
    for key,val in the_dict.items():
        map_key = key
        map_val = val
        map_element = (map_key, map_val)
        my_map.insert(map_element)
    return my_map

def array_to_vector(arr):
    # arr is a 1d numpy array of strings
    cdef string element
    cdef int i
    cdef vector[string] my_vector
    for i in range(arr.shape[0]):
        element = arr[i]
        my_vector.push_back(element)
    return my_vector

def pivot(df, index, column, value):
    """
    A very basic and limited, but hopefully fast implementation of pivot table.
    Aggregates by sum.
    Arguments:
    df: pandas dataframe
    index: string, name of column that you want to become the index. values must be of type string.
    column: string, name of column that contains as values the columns of the pivot table. values must be of type string.
    value: string, name of column that contains as values the values of the pivot table. values must be of type np.float64.
    Returns a pandas dataframe
    """
    idx_list = list(df[index].unique())
    col_list = list(df[column].unique())
    idx_dict = {idx_list[i]:i for i in range(len(idx_list))}
    col_dict = {col_list[i]:i for i in range(len(col_list))}
    idx = dict_to_cmap(idx_dict)
    col = dict_to_cmap(col_dict)
    idx_arr = array_to_vector(df[index].to_numpy())
    col_arr = array_to_vector(df[column].to_numpy())
    pivot_arr = pivot_cython(idx_arr, col_arr, df[value].to_numpy(), idx, col)
    arr = np.array(pivot_arr)
    pivot_df = pd.DataFrame(arr, index=idx_list, columns=col_list)
    return pivot_df

cdef double[:, :] pivot_cython(vector[string] idx_arr, vector[string] col_arr, double[:] value_arr, cmap[string, int] idx, cmap[string, int] col):
    cdef int N = idx.size()
    cdef int M = col.size()
    cdef double[:, :] pivot_arr = np.zeros((N, M), dtype=np.float64)
    cdef int i
    cdef int j
    cdef int k
    cdef double value
    for k in range(idx_arr.size()):
        i = idx[idx_arr[k]]
        j = col[col_arr[k]]
        value = value_arr[k]
        pivot_arr[i, j] += value
    return pivot_arr
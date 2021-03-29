# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

#from libcpp.string cimport string
#from libcpp.map cimport map as cmap
#from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport sqrt
from libc.math cimport isnan
#from cython.operator import dereference, postincrement
#from libc.stdlib cimport malloc
from libcpp.algorithm cimport sort as stdsort
import pandas as pd
import numpy as np
cimport numpy as np
import time

# TODO: median slow when dupes
# TODO: faster dropna, fillna?
# TODO: product function?
# TODO: further optimization?
# TODO: try multithread when values or aggfunc multiple? or even in general?
def pivot_table(df, index, columns, values, aggfunc='mean', fill_value=None, dropna=True, dropna_idxcol=True):
    """
    Summary:
        A limited, but hopefully fast implementation of pivot table.
        Tends to be faster than pandas.pivot_table when resulting pivot table is sparse.
        The main limitation is that you must include index, columns, values and you must aggregate.
        You also must aggregate by a list of preconstructed functions:
            For numerical values (np.float64 or np.int64), you can aggregate by any of 
                ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
            For other values, you can aggregate by any of 
                ['count', 'nunique']

        The arguments and return format mimic pandas very closely, with a few small differences:
        1) Occasionaly the ordering of the columns will be different, such as when passing a list of aggfuncs 
            with a single value column
        2) When passing values of type np.int64, values of type np.float64 will be returned. 
            Pandas returns np.int64 in some cases and np.float64 in others.
        3) When passing multi index or column, pandas constructs the cartesion product space, whereas this pivot constructs the 
            subspace of the product space where the tuples exist in the passed dataframe.
        4) The following arguments are not supported here: margins, margins_name, observed.

        Generally on a dataframe with many rows and many distinct values in the passed index and column, the performance of 
            this pivot_table function beats pandas significantly, by a factor of 2 to 20.
        On a dataframe with many rows but few distinct values in the passed index and column, the speed of this pivot_table
            tends to be roughly on par with pandas, and in some cases can actually be slower.

    Arguments:
        df (pandas dataframe)
        index (string or list): name(s) of column(s) that you want to become the index of the pivot table. 
        columns (string or list): name(s) of column(s) that contains as values the columns of the pivot table. 
        values (string or list): name(s) of column(s) that contains as values the values of the pivot table.
        aggfunc (string, list, or dict, default 'mean'): name of aggregation function. must be on implemented list above.
            if aggfunc is a dict, the format must be as in the following example:
            values = ['column_name1', 'column_name2', 'column_name3']
            aggfunc = {'column_name1': 'mean', 'column_name2': 'median', 'column_name3': ['count', 'nunique']}
        fill_value (scalar, default None): value to replace missing values with in the pivot table.
        dropna (bool, default True): if True rows and columns that are entirely NaN values will be dropped.
        dropna_idxcol (bool, default True): if True rows where the passed index or column contain NaNs will be dropped.
            if False, NaN will be given its own index or column when appropriate.

    Returns:
        pivot_df (pandas dataframe)
    """
    
    # tick = time.perf_counter()
    assert isinstance(index, str) or isinstance(index, list)
    assert isinstance(columns, str) or isinstance(columns, list)
    if isinstance(index, str):
        if dropna_idxcol:
            df = df.dropna(subset=[index])
    if isinstance(index, list):
        if dropna_idxcol:
            df = df.dropna(subset=index)
        if len(index) == 1:
            index = index[0]
    if isinstance(columns, str):
        if dropna_idxcol:
            df = df.dropna(subset=[columns])
    if isinstance(columns, list):
        if dropna_idxcol:
            df = df.dropna(subset=columns)
        if len(columns) == 1:
            columns = columns[0]

    if isinstance(index, str):
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = df[index].factorize(sort=True, na_sentinel=None)
        #print('factorize idx', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        idx_arr, idx_arr_unique = pd.MultiIndex.from_frame(df[index]).factorize(sort=True, na_sentinel=None)
        idx_arr_unique = pd.MultiIndex.from_tuples(idx_arr_unique, names=index)
        #print('factorize idx', time.perf_counter() - tick1)
    if isinstance(columns, str):
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = df[columns].factorize(sort=True, na_sentinel=None)
        #print('tuple conversion col', time.perf_counter() - tick1)
    else: #TODO: any speedup here?
        #tick1 = time.perf_counter()
        col_arr, col_arr_unique = pd.MultiIndex.from_frame(df[columns]).factorize(sort=True, na_sentinel=None)
        col_arr_unique = pd.MultiIndex.from_tuples(col_arr_unique, names=columns)
        #print('factorize col', time.perf_counter() - tick1)
    # print('prepare index and columns', time.perf_counter() - tick)

    # tick = time.perf_counter()
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
    # print('compute pivot table', time.perf_counter() - tick)
    
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
    pivot_arr = pivot_compute_agg(aggfunc, fill_value, idx_arr, col_arr, values_series, n_idx, n_col)
    pivot_df = pd.DataFrame(pivot_arr, index=idx_arr_unique, columns=col_arr_unique)
    pivot_df = pivot_drop_fill(aggfunc, fill_value, dropna, pivot_df, idx_arr, col_arr, values_series, n_idx, n_col)
    pivot_df.index.rename(index, inplace=True)
    pivot_df.columns.rename(columns, inplace=True)

    return pivot_df

def pivot_compute_agg(aggfunc, fill_value, idx_arr, col_arr, values_series, n_idx, n_col):
    # computes pivot table and aggregates.
    # if fill_value is zero then sum, count, and nunique fill by zero right away since these aggfuncs never have
    # to consider dropna. 
    # in all other cases (other aggfuncs, or all other fill_value) NaNs are returned when appropriate and filled later
    # after dropna has been called

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
            pivot_arr = pivot_cython_sum(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, fill_value == 0)
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
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col, nans_arr, fill_value == 0)
        elif aggfunc == 'median':
            pivot_arr = pivot_cython_agg(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, median_cython_sort)
        elif aggfunc == 'nunique':
            pivot_arr = pivot_cython_agg_nan(idx_arr, col_arr, values_series.to_numpy(), n_idx, n_col, nunique_cython, fill_value == 0)
    else:
        if aggfunc == 'count':
            nans_arr = values_series.isna().to_numpy()
            pivot_arr = pivot_cython_count(idx_arr, col_arr, n_idx, n_col, nans_arr, fill_value == 0)
        elif aggfunc == 'nunique':
            nans_arr = values_series.isna().to_numpy()
            values_arr, _ = values_series.factorize(na_sentinel=None)
            values_arr = values_arr.astype(np.float64)
            values_arr[nans_arr] = np.nan
            pivot_arr = pivot_cython_agg_nan(idx_arr, col_arr, values_arr, n_idx, n_col, nunique_cython, fill_value == 0)
    arr = np.array(pivot_arr)

    return arr

def pivot_drop_fill(aggfunc, fill_value, dropna, pivot_df, idx_arr, col_arr, values_series, n_idx, n_col):
    if aggfunc in ['sum', 'count', 'nunique']:
        # these functions can only have nans if (idx, col) doesn't exist so no need to drop. only fill.
        # if fill_value == 0, NaNs have already been filled by pivot_compute_agg
        if fill_value is not None and fill_value is not np.nan and fill_value != 0:
            #tick = time.perf_counter()
            pivot_df = pivot_df.fillna(fill_value)
            #print('fillna', time.perf_counter() - tick)
    elif aggfunc in ['mean', 'max', 'min', 'median']:
        # these functions can have nans if (idx, col) doesn't exist or if (idx, col) has only NaNs.
        # must check dropping in this case
        if dropna and values_series.isna().to_numpy().any(): # TODO speedup
            #tick = time.perf_counter()
            pivot_df = pivot_df.dropna(axis=0, how='all')
            pivot_df = pivot_df.dropna(axis=1, how='all')
            #print('dropna', time.perf_counter() - tick)
        if fill_value is not None and fill_value is not np.nan:
            #tick = time.perf_counter()
            pivot_df = pivot_df.fillna(fill_value)
            #print('fillna', time.perf_counter() - tick)
    elif aggfunc == 'std':
        # this function can have nans if (idx, col) doesn't exist or if (idx, col) has only 0 or 1 non NaN value.
        # must check dropping in this case
        if dropna: # TODO speedup
            #tick = time.perf_counter()
            pivot_df = pivot_df.dropna(axis=0, how='all')
            pivot_df = pivot_df.dropna(axis=1, how='all')
            #print('dropna', time.perf_counter() - tick)
        if fill_value is not None and fill_value is not np.nan:
            #tick = time.perf_counter()
            pivot_df = pivot_df.fillna(fill_value)
            #print('fillna', time.perf_counter() - tick)

    return pivot_df

cdef double[:, :] pivot_cython_sum(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, bool fill_zero):
    init = np.zeros((N, M), dtype=np.float64)
    if not fill_zero:
        init.fill(np.nan)
    cdef double[:, :] pivot_arr = init
    cdef int i, j, k
    cdef double value
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        value = value_arr[k]
        #####################################
        # pandas considers empty sums to be 0
        if isnan(pivot_arr[i, j]):
            pivot_arr[i, j] = 0.0
        #####################################
        if not isnan(value):
            pivot_arr[i, j] += value
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

# cdef double[:, :] pivot_cython_std_welford(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
#     nans = np.zeros((N, M), dtype=np.float64)
#     nans.fill(np.nan)
#     cdef double[:, :] pivot_arr = nans
#     cdef double[:, :] mean_arr = np.zeros((N, M), dtype=np.float64)
#     cdef double[:, :] pivot_counts_arr = np.zeros((N, M), dtype=np.float64)
#     cdef int i, j, k
#     cdef double value
#     cdef double new_mean
#     cdef double divisor
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         if not isnan(value_arr[k]):
#             if not isnan(pivot_arr[i, j]):
#                 value = value_arr[k]
#                 pivot_counts_arr[i, j] += 1.0
#                 new_mean = mean_arr[i, j] + (value - mean_arr[i, j]) / pivot_counts_arr[i, j]
#                 pivot_arr[i, j] += (value - mean_arr[i, j]) * (value - new_mean)
#                 mean_arr[i, j] = new_mean
#             else:
#                 value = value_arr[k]
#                 pivot_counts_arr[i, j] += 1.0
#                 pivot_arr[i, j] = 0
#                 mean_arr[i, j] = value                
#     for i in range(N):
#         for j in range(M):
#             divisor = pivot_counts_arr[i, j]
#             if divisor != 0.0:
#                 pivot_arr[i, j] /= (divisor - 1)
#                 pivot_arr[i, j] = sqrt(pivot_arr[i, j])
#             else:
#                 pivot_arr[i, j] = np.nan
#     return pivot_arr

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

cdef double[:, :] pivot_cython_count(long[:] idx_arr, long[:] col_arr, int N, int M, bool[:] nans_arr, bool fill_zero):
    init = np.zeros((N, M), dtype=np.float64)
    if not fill_zero:
        init.fill(np.nan)
    cdef double[:, :] pivot_arr = init
    cdef int i, j, k
    for k in range(idx_arr.shape[0]):
        i = idx_arr[k]
        j = col_arr[k]
        if isnan(pivot_arr[i, j]):
            pivot_arr[i, j] = 0.0
        if not nans_arr[k]:
            pivot_arr[i, j] += 1.0
    return pivot_arr

# cdef bool[:, :] find_missing_cython(long[:] idx_arr, long[:] col_arr, int N, int M):
#     cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
#     cdef int i, j, k
#     cdef double value
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         missing_arr[i, j] = 0
#     return missing_arr

# cdef bool[:, :] find_missing_cython_0(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
#     cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
#     cdef int i, j, k
#     cdef double value
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         if not isnan(value_arr[k]):
#             missing_arr[i, j] = 0
#     return missing_arr

# cdef bool[:, :] find_missing_cython_1(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M):
#     cdef bool[:, :] missing_arr = np.ones((N, M), dtype=np.bool)
#     cdef bool[:, :] exists_arr = np.zeros((N, M), dtype=np.bool)
#     cdef int i, j, k
#     cdef double value
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         if not isnan(value_arr[k]):
#             if exists_arr[i, j]:
#                 missing_arr[i, j] = 0
#             else:
#                 exists_arr[i, j] = 1
#     return missing_arr

ctypedef double vec_to_double(vector[double] &)
#ctypedef long vec_to_long(vector[double] &)

cdef double sum_cython(vector[double] &vec):
    cdef int k
    cdef double value = 0.0
    for k in range(vec.size()):
        value += vec[k]
    return value

cdef double median_cython_sort(vector[double] &vec):
    cdef int k
    cdef int idx
    cdef double value = 0.0
    cdef double med
    stdsort(vec.begin(), vec.end())
    idx = vec.size() / 2
    if vec.size() % 2 == 1:
        med = vec[idx]
    elif vec.size() == 0:
        med = np.nan
    else:
        med = (vec[idx] + vec[idx-1]) / 2
    return med

# very slow on duplicates
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
        #print(k)
        if k < mid:
            i = k + 1
        elif k > mid:
            j = k - 1
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
            if vec[j] == x:
                i += 1

cdef double max_left_cython(vector[double] & vec, int upper_idx):
    # never called on empty vec
    cdef int i
    cdef double value
    value = vec[0]
    for i in range(1, upper_idx + 1):
        if value < vec[i]:
            value = vec[i]
    return value

cdef double nunique_cython(vector[double] &vec):
    cdef int k
    cdef cset[double] value_set
    cdef double n
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

cdef double[:, :] pivot_cython_agg_nan(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_double agg, bool fill_zero):
    #tick = time.perf_counter()
    cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
    cdef vector[double].iterator it
    init = np.zeros((N, M), dtype=np.float64)
    if not fill_zero:
        init.fill(np.nan)
    cdef double[:, :] pivot_arr_return = init
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
            pivot_arr_return[i, j] = 0.0
    #print('push_back', time.perf_counter() - tick)
    #tick = time.perf_counter()
    for i in range(N):
        for j in range(M):
                value = agg(pivot_arr[i*M + j])
                if value != 0:
                    pivot_arr_return[i, j] = value
    #print('agg and assign', time.perf_counter() - tick)
    return pivot_arr_return

# cdef long[:, :] pivot_cython_agg_int(long[:] idx_arr, long[:] col_arr, double[:] value_arr, int N, int M, vec_to_long agg):
#     #tick = time.perf_counter()
#     cdef vector[vector[double]] pivot_arr = vector[vector[double]](N*M)
#     cdef vector[double].iterator it
#     cdef long[:, :] pivot_arr_return = np.zeros((N, M), dtype=np.int64)
#     cdef int i, j
#     cdef double value
#     cdef long value_int
#     #print('cdef and initialize', time.perf_counter() - tick)
#     #tick = time.perf_counter()
#     for k in range(idx_arr.shape[0]):
#         i = idx_arr[k]
#         j = col_arr[k]
#         value = value_arr[k]
#         if not isnan(value):
#             pivot_arr[i*M + j].push_back(value)
#     #print('push_back', time.perf_counter() - tick)
#     #tick = time.perf_counter()
#     for i in range(N):
#         for j in range(M):
#                 value_int = agg(pivot_arr[i*M + j])
#                 pivot_arr_return[i, j] = value_int
#     #print('agg and assign', time.perf_counter() - tick)
#     return pivot_arr_return

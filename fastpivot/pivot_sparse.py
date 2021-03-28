import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import time

# cython version actual beats this in many cases, surprisingly
def pivot_sparse(df, index, columns, values, fill_value=None, dropna_idxcol=True, as_pd=True):
    """
    Summary:
        Uses scipy.sparse.coo_matrix to construct a pivot table.
        This uses less memory and is faster in most cases when the resulting pivot_table will be sparse.
        Aggregates by sum. Less functionality overall, but efficient for its usecase.

    Arguments:
        df (pandas dataframe)
        index (string or list): name(s) of column(s) that you want to become the index of the pivot table. 
        columns (string or list): name(s) of column(s) that contains as values the columns of the pivot table. 
        values (string): name of column that contains as values the values of the pivot table.
        fill_value (scalar, default None): value to replace missing values with in the pivot table.
        dropna_idxcol (bool, default True): if True rows where the passed index or column contain NaNs will be dropped. 
            if False, NaN will be given its own index or column when appropriate.
        as_pd (bool, default True): if True returns pandas dataframe. if false, returns the scipy coo matrix (unaggregated), 
            the index array, and the column array separately. In this case, fill_value is ignored
    
    Returns:
        pivot_df (pandas dataframe)
        ---OR---
        coo (scipy coo matrix): unaggregated.
        idx_labels (pandas index or multiindex): contains distinct index values for pivot table.
        col_labels (pandas index or multiindex): contains distinct column values for pivot table.
            if the coo matrix contains a value at pair (i, j) then the index label is idx_labels[i] and the column label 
            is col_labels[j].
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
    coo = coo_matrix((df[values], (idx_arr, col_arr)), shape=(idx_arr_unique.shape[0], col_arr_unique.shape[0]))
    # print('coo', time.perf_counter() - tick)

    #print(coo.toarray().shape)
    #print(coo.toarray())

    if as_pd:
        # tick = time.perf_counter()
        pivot_df = pd.DataFrame.sparse.from_spmatrix(coo, index=idx_arr_unique, columns=col_arr_unique)
        pivot_df.index.rename(index, inplace=True)
        pivot_df.columns.rename(columns, inplace=True)
        #pivot_df = pd.DataFrame(coo.toarray(), index=idx_arr_unique, columns=col_arr_unique)
        # print('from_spmatrix', time.perf_counter() - tick)

        if fill_value is not None and fill_value is not np.nan:
            # tick = time.perf_counter()
            pivot_df = pivot_df.fillna(fill_value)
            # print('fillna', time.perf_counter() - tick)

        return pivot_df

    else:

        return coo, idx_arr_unique, col_arr_unique
# FastPivot
A basic but fast reconstruction of pandas.pivot_table

Contains two functions (see documentation below):

fastpivot.pivot_table 

fastpivot.pivot_sparse

# Installation

pip install fastpivot

Latest version: 0.1.12

# Documentation

~~~text
fastpivot.pivot_table(df, index, columns, values, aggfunc='mean', fill_value=None, dropna=True, dropna_idxcol=True):
    A limited, but hopefully fast implementation of pivot table.
    Tends to be faster than pandas.pivot_table when resulting pivot table is sparse.
    The main limitation is that you must include index, columns, values and you must aggregate.
    You also must aggregate by a list of preconstructed functions:
        For numerical values (np.float64 or np.int64), you can aggregate by any of 
           ['sum', 'mean', 'std', 'max', 'min', 'count', 'median', 'nunique']
        For other values, you can aggregate by any of 
            ['count', 'nunique']

    The arguments and return format mimic pandas very closely, with a few small differences:
    1) Occasionaly the ordering of the columns will be different, such as when passing a list of aggfuncs with a single value column
    2) When passing values of type np.int64, values of type np.float64 will be returned. Pandas returns np.int64 in some cases and np.float64 in others.
    3) When passing multi index or column, pandas constructs the cartesion product space, whereas this pivot constructs the subspace of the product space where the tuples exist in the passed dataframe.
    4) The following arguments are not supported here: margins, margins_name, observed.

    Generally on a dataframe with many rows and many distinct values in the passed index and column, the performance of this pivot_table function beats pandas significantly (see benchmarks)
    On a dataframe with many rows but few distinct values in the passed index and column, the speed of this pivot_table tends to be roughly on par with pandas, and in some cases can actually be slower.

    Arguments:
    df (required): pandas dataframe
    index (required): string or list, name(s) of column(s) that you want to become the index of the pivot table. 
    columns (required): string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values (required): string or list, name(s) of column(s) that contains as values the values of the pivot table.
    aggfunc (default 'mean'): string, list, or dict, name of aggregation function. must be on implemented list above.
        if aggfunc is a dict, the format must be as in the following example:
        values = ['column_name1', 'column_name2', 'column_name3']
        aggfunc = {'column_name1': 'mean', 'column_name2': 'median', 'column_name3': ['count', 'nunique']}
    fill_value (default None): scalar, value to replace missing values with in the pivot table.
    dropna (default True): bool, if True rows and columns that are entirely NaN values will be dropped.
    dropna_idxcol (default True): bool, if True rows where the passed index or column contain NaNs will be dropped.
        if False, NaN will be given its own index or column when appropriate.

    Returns:
    pivot_df: pandas dataframe

fastpivot.pivot_sparse(df, index, columns, values, fill_value=None, dropna_idxcol=True):
    Uses scipy.sparse.coo_matrix to construct a pivot table.
    This uses less memory and is faster in most cases when the resulting pivot_table will be sparse.
    Aggregates by sum. Less functionality overall, but efficient for its usecase.
    
    Arguments:
    df (required): pandas dataframe
    index (required): string or list, name(s) of column(s) that you want to become the index of the pivot table. 
    columns (required): string or list, name(s) of column(s) that contains as values the columns of the pivot table. 
    values (required): string, name of column that contains as values the values of the pivot table.
    fill_value (default None): scalar, value to replace missing values with in the pivot table.
    dropna_idxcol (default True): bool, if True rows where the passed index or column contain NaNs will be dropped.
        if False, NaN will be given its own index or column when appropriate.
    
    Returns:
    pivot_df: pandas dataframe
~~~

# Example
~~~text
import pandas as pd
import numpy as np
import time
from fastpivot import pivot_table, pivot_sparse 

N_ROWS = 100000
N_COLS = 1000
N_IDX = 1000

NAME_IDX = 'pivot_idx'
NAME_COL = 'pivot_col'
NAME_VALUE = 'value'

col1 = ['idx{}'.format(x) for x in np.random.randint(0, N_IDX, size=N_ROWS)]
col2 = ['col{}'.format(x) for x in np.random.randint(0, N_COLS, size=N_ROWS)]
col3 = [x for x in np.random.normal(size=N_ROWS)]

data = np.transpose([col1, col2, col3])
df = pd.DataFrame(data, columns=[NAME_IDX, NAME_COL, NAME_VALUE], index=range(len(data)))
df[NAME_VALUE] = df[NAME_VALUE].astype(np.float64)
print(df)

msg = 'fastpivot.pivot_table'
tick = time.perf_counter()
pivot_fast = pivot_table(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
print(msg, time.perf_counter() - tick)
print(pivot_fast)

msg = 'pandas.pivot_table'
tick = time.perf_counter()
pivot_pandas = df.pivot_table(index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0, aggfunc='sum')
print(msg, time.perf_counter() - tick)
print(pivot_pandas)

msg = 'fastpivot.pivot_sparse'
tick = time.perf_counter()
pivot_sparse_df = pivot_sparse(df, index=NAME_IDX, columns=NAME_COL, values=NAME_VALUE, fill_value=0.0)
print(msg, time.perf_counter() - tick)
print(pivot_sparse_df)
~~~
# Benchmarks

Computed using AMD Ryzen 5 2600 CPU and 16 GB RAM.  
Time is in denoted in seconds.  
NaN indicates that the function is not capable of doing the computation. OOM indicates that the function ran out of memory.  

N_ROWS denotes the number of rows in the input dataframe.  
N_COLS denotes the number of distinct values in the column which is passed as the columns argument.  
N_IDX denotes the number of distinct values in the column which is passed as the index argument.  

For multicol and multiidx, both columns for the columns or index argument have N_COLS or N_IDX distinct values respectively.  
All examples use dtype np.float64 and fill_value = 0.  
All arguments except for df, index, column, values, aggfunc, and fill_value are default.  

~~~text

Test 1:

N_ROWS = 100000
N_COLS = 1000
N_IDX = 1000

              fastpivot.pivot_table  pandas.pivot_table  fastpivot.pivot_sparse
sum                        0.060811            0.242244                0.067733
mean                       0.047007            0.219768                     NaN
std                        0.062170            0.041423                     NaN
max                        0.042843            0.223578                     NaN
min                        0.041605            0.225407                     NaN
count                      0.037413            0.223480                     NaN
nunique                    0.080111            0.258028                     NaN
median                     0.095170            0.216555                     NaN
multicol sum               0.793830           17.408360                4.243026
multiidx sum               0.978207            4.674374                0.307516

Winner: fastpivot.pivot_table

Test 2:

N_ROWS = 1000000
N_COLS = 10
N_IDX = 10

              fastpivot.pivot_table  pandas.pivot_table  fastpivot.pivot_sparse
sum                        0.317188            0.116396                0.317928
mean                       0.267151            0.102374                     NaN
std                        0.275409            0.101728                     NaN
max                        0.266737            0.118855                     NaN
min                        0.271199            0.113987                     NaN
count                      0.279253            0.115915                     NaN
nunique                    0.306899            0.267046                     NaN
median                     0.289684            0.135484                     NaN
multicol sum               0.847213            0.160205                0.831495
multiidx sum               0.822259            0.150282                0.883725

Winner: pandas.pivot_table

Test 3:

N_ROWS = 2000000
N_COLS = 1000
N_IDX = 50000

              fastpivot.pivot_table  pandas.pivot_table  fastpivot.pivot_sparse
sum                        1.178474            3.889266                0.844055
mean                       1.608712            3.877892                     NaN
std                        1.892230            1.196348                     NaN
max                        1.295431            3.850022                     NaN
min                        1.288993            3.868061                     NaN
count                      1.117461            3.870997                     NaN
nunique                    3.148521            4.536493                     NaN
median                     3.951064            3.971369                     NaN
multicol sum                    OOM                 OOM               38.984358
multiidx sum                    OOM                 OOM               12.011507

Winners: fastpivot.pivot_sparse (when applicable), fastpivot.pivot_table

Test 4: 

N_ROWS = 1000000
N_COLS = 1000
N_IDX = 100

              fastpivot.pivot_table  pandas.pivot_table  fastpivot.pivot_sparse
sum                        0.344388            0.303900                0.364910
mean                       0.281136            0.292918                     NaN
std                        0.293342            0.161475                     NaN
max                        0.284871            0.296082                     NaN
min                        0.280373            0.285619                     NaN
count                      0.274774            0.269245                     NaN
nunique                    0.397173            0.500805                     NaN
median                     0.356431            0.303541                     NaN
multicol sum               3.395306           89.057448               29.051697
multiidx sum               0.737902            1.263041                0.946133

Winner: fastpivot.pivot_table

Test 5:

N_ROWS = 1000000
N_COLS = 1000
N_IDX = 10000

              fastpivot.pivot_table  pandas.pivot_table  fastpivot.pivot_sparse
sum                        0.425536            1.233995                0.392080
mean                       0.480641            1.189147                     NaN
std                        0.607801            0.488366                     NaN
max                        0.412936            1.213504                     NaN
min                        0.414011            1.186172                     NaN
count                      0.379848            1.199575                     NaN
nunique                    1.043115            1.473638                     NaN
median                     1.177022            1.258532                     NaN
multicol sum                    OOM                 OOM               30.198564
multiidx sum                    OOM                 OOM                4.907232

Winners: fastpivot.pivot_sparse (when applicable), fastpivot.pivot_table

~~~

Takeaway:

pandas.pivot_table is highly flexible and optimized for input with a large number of rows but few distinct values in the index and column, so that the resulting pivot table is small and each value results from aggregating a large number of values in the original dataframe.  

fastpivot.pivot_table is reasonably flexible (though not as flexible as pandas) and outperforms pandas when there are a large number of distinct values in the index and column. It outperforms pandas especially for multi column and mutli index input.  

fastpivot.pivot_sparse is very limited but faster and much more memory efficient than pandas when there is an extremely large number of distinct values in the index and column, so that the resulting dataframe is extremely sparse.  

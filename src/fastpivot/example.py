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
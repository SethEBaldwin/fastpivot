import pivot
import pandas as pd
import numpy as np
import time

column1 = 'to_be_idx'
column2 = 'to_be_col'
column3 = 'value'

row1 = ['111', 'sub2', 7]
row2 = ['222', 'sub2', 2]
row3 = ['222', 'sub1', 11]
data = [row1, row2, row3]
df = pd.DataFrame(data, columns=[column1, column2, column3], index=range(len(data)))
df[column3] = df[column3].astype(np.float64)
print(df)

msg = 'cython'
tick = time.perf_counter()
pivot_cython = pivot.pivot(df, index=column1, column=column2, value=column3)
print(msg, time.perf_counter() - tick)
print(pivot_cython)

msg = 'pandas'
tick = time.perf_counter()
pivot_pandas = df.pivot_table(index=column1, columns=[column2], values=column3, fill_value=0)
print(msg, time.perf_counter() - tick)
print(pivot_pandas)

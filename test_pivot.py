import pivot
import pandas as pd
import numpy as np
import time

# create data
    
column1 = 'to_be_idx'
column2 = 'to_be_col'
column3 = 'value'

n_rows = 100
n_cols = 10
n_idx = 10
col1 = ['idx{}'.format(x) for x in np.random.randint(0, n_idx, size=n_rows)]
col2 = ['col{}'.format(x) for x in np.random.randint(0, n_cols, size=n_rows)]
col3 = [x for x in np.random.normal(size=n_rows)]

data = np.transpose([col1, col2, col3])
df = pd.DataFrame(data, columns=[column1, column2, column3], index=range(len(data)))
df[column3] = df[column3].astype(np.float64)
#print(df)

# time

msg = 'cython'
tick = time.perf_counter()
pivot_cython = pivot.pivot(df, index=column1, column=column2, value=column3)
print(msg, time.perf_counter() - tick)
#print(pivot_cython)

msg = 'pandas'
tick = time.perf_counter()
pivot_pandas = df.pivot_table(index=column1, columns=[column2], values=column3, fill_value=0.0, aggfunc='sum')
print(msg, time.perf_counter() - tick)
#print(pivot_pandas)

# check results are equal

print('componentwise equal: ', (pivot_cython.to_numpy() == pivot_pandas.to_numpy()).all())
epsilon = 1e-8
print('componentwise within {} :'.format(epsilon), (np.absolute(pivot_cython.to_numpy() - pivot_pandas.to_numpy()) < epsilon).all())

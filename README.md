# FastPivot
A basic but fast reconstruction of pandas.pivot_table

# Installation

TODO

# Example

TODO

# Benchmarks

Computed using AMD Ryzen 5 2600 CPU and 16 GB RAM.  
Time is in denoted in seconds.  
NaN indicates either the function is not capable of doing the computation or the function ran out of memory.  

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
multicol sum                    NaN                 NaN               38.984358
multiidx sum                    NaN                 NaN               12.011507

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
multicol sum                    NaN                 NaN               30.198564
multiidx sum                    Nan                 NaN                4.907232

Winners: fastpivot.pivot_sparse (when applicable), fastpivot.pivot_table

~~~

Takeaway:

pandas.pivot_table is highly flexible and optimized for input with a large number of rows but few distinct values in the index and column, so that the resulting pivot table is small and each value results from aggregating a large number of values in the original dataframe.  

fastpivot.pivot_table is reasonably flexible (though not as flexible as pandas) and outperforms pandas when there are a large number of distinct values in the index and column. It outperforms pandas especially for multi column and mutli index input.  

fastpivot.pivot_sparse is very limited but faster and much more memory efficient than pandas when there is an extremely large number of distinct values in the index and column, so that the resulting dataframe is extremely sparse.  

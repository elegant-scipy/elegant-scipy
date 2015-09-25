
# NumPy Basics

## Numpy N-dimensional arrays

### Why use ndarrays as opposed to Python lists?

### Broadcasting

### NumPy functions to cover
np.transpose
np.log2
np.mean
np.sort
np.round

## from scipy import stats
stats.rankdata

the nieve way to do it is using t tests, using the stats package. But there are better ways to use it in R

## Quantile normalization with NumPy and SciPy

### Expression data
Explain briefly about expression and microarrays

Import a dataset (e.g. ALL?)

Explore the dataset


Given an expression matrix (microarray data, read counts, etc) of ngenes by nsamples, quantile normalization ensures all samples have the same spread of data (by construction). It involves:

(optionally) log-transforming the data
sorting all the data points column-wise
averaging the rows
replacing each column quantile with the quantile of the average column.
This can be done with numpy and scipy easily and efficiently. Assume we've read in the input matrix as X:

```python
# import the goodies
import numpy as np
from scipy import stats

# log-transform the data
logX = np.log2(X + 1)

# compute the quantiles
log_quantiles = np.mean(np.sort(logX, axis=0), axis=1)

# compute the column-wise ranks; need to do a round-trip through list
ranks = np.transpose([np.round(stats.rankdata(col)).astype(int) for col in X.T])
# alternative: ranks = np.argsort(np.argsort(X, axis=0), axis=0)

# index the quantiles for each rank with the ranks matrix
logXn = log_quantiles[ranks]

# convert the data back to counts (casting to int is optional)
Xn = np.round(2**logXn - 1).astype(int)
```

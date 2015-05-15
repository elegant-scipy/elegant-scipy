
# 2D Array Manipulation for RNAseq

## Expression data

Explain briefly about expression and microarrays

Import a tiny toy example


## Numpy N-dimensional arrays

One of the key NumPy datatypes is the N-dimensional array (ndarray).

All items in an ndarray must be the same type and size.

Ndarrays are called N-dimentional because they can have any number of
dimensions.
For example a 1-dimesional array would look like this: [picture]

A 2-dimensional array, for example a 2 by 3 array: [picture]

101

001

A 3-dimensional array: [picture]


A 4 dimensional array becomes tricky to visualise,
so perhaps it is easier to think of it in terms of a use case.
Let's say you have an ndarray that describes an object over time.
You would need three dimensions to describe the position of
the objects and a fourth to indicate time.

```python
    # Make ndarrays of various dimensions
```

```python
    # Use .shape to check their dimensions
```

### Why use ndarrays as opposed to Python lists?

Ndarrays are fast because vectorized oporations can be performed on them.
Let's say you have a list and you want to multiple every element in the list by 5.
A standard Python approach would be to write a loop that iterates through the
elements of the list and multiples each one by 5.
However, if your data were instead represented as an ndarray,
you can multiple every element in the ndarray by 5 simultaneously.

```python
    import numpy as np

    # Create an ndarray of integers in the range 0 up to (but not including) 10,000,000
    nd_array = np.arange(1e6)
    # Convert arr to a list
    list_array = nd_array.tolist()
```

```python
    %%timeit # Use the Ipython "magic" command timeit to time how long it takes to multiply each element in the ndarray by 5
    x = nd_array * 5
```

```python
    %%timeit # Time how long it takes to multiply each element in the list by 5
    for i, val in enumerate(list_array):
        list_array[i] = val * 5
```

Ndarrays are also size effiecient.
In Python, each element in a list as an object and is given a health memory allocation.
In contrast, for ndarrays you
(or in this case the arange function)
decide how much memory to allocate to the elements.
This is generally much less than would be given to objects in a python list.

Ndarrays contain pointers to other ndarrays or even other python data types such
as ...
This means that you can have a single copy of your data and access subsets via
other variables without duplicating that portion of your data.
However, this also means that if you may inadvertently edit your data set
when you might think you are making a copy of it.

```python
    # Create an ndarray x
    x = np.array([1, 2, 3], np.int32)
    print(x)
```

```python
    # Create variable name y that points to the first two values in x
    y = x[:2]
    print(y)
```

```python
    # Set the first element of y to be 6
    y[0] = 6
    print(y)
```

```python
    # Now the first element in x has changed to 6!
    print(x)
```

```python
    # If you actually wanted a copy of your ndarray use it's copy method
    y = x[:2].copy()
```

### Broadcasting

Broadcasting is ...

Broadcasting relates to vectorization ... see
http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

## Exploring (some simple descriptive statistics and plots PCA/MDS?)

Import a full dataset

Data set:

- Hapmap RNA seq data set
- Do Male vs. female analysis
- This is explained in the limma voom documentation:  
http://www.bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf
- Here are the gene counts:
http://eqtl.uchicago.edu/RNA_Seq_data/results/final_gene_counts.gz
- RNAseq reads also available here (for streaming chapter):
http://eqtl.uchicago.edu/RNA_Seq_data/unmapped_reads/

```python
import urllib
import gzip
import numpy as np
import sys

url = "http://eqtl.uchicago.edu/RNA_Seq_data/results/final_gene_counts.gz"

remote_filehandle = urllib.request.urlopen(url)
with gzip.open(remote_filehandle, 'rt') as f:
    all_lines = []
    for line in f:
        line_array = np.array(line.split()[3:]) # [3:] to remove first three cols which are gene, chr, len
        all_lines.append(line_array)
    data = np.asarray(all_lines)

    data = data[1:] # remove header row
    data = np.array(data, dtype='int')
    print(data)
```

Explore the dataset
```python
# bar plot
%matplotlib inline

small_data = data[:10] # reduce size of data

from pylab import *
figure()
boxplot(small_data)
show()

```

### NumPy/SciPy functions to cover
np.transpose
np.log2
np.mean
np.sort
np.round

```python
from scipy import stats
stats.rankdata
```

## Differential expression analysis

Explain briefly about differential expression analysis

the nieve way to do it is using t tests, using the stats package.
But there are better ways to use it in R

### Normalization

Why normalise? Show some boxplots.

An example of the types of plots I'd like to show:
http://www.nature.com/nbt/journal/v32/n9/images_article/nbt.2931-F2.jpg


## Quantile normalization with NumPy and SciPy

Given an expression matrix (microarray data, read counts, etc) of ngenes by nsamples, quantile normalization ensures all samples have the same spread of data (by construction). It involves:

(optionally) log-transforming the data
sorting all the data points column-wise
averaging the rows
replacing each column quantile with the quantile of the average column.
This can be done with numpy and scipy easily and efficiently.
Assume we've read in the input matrix as X:

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


## Best-Practice Differential Expression Analysis: Accessing R statistical packages with RPy2

Use rpy to call limma

- limma takes non-count data (e.g. if you have already normalised it.)
- use edgeR if you have count data (i.e. whole numbers)
- limma-voom is useful if you haven’t already normalised by library size

Diagnostic plots

- P-value histogram
- Volcano plot of results


# 2D Array Manipulation for RNAseq

## Gene Expression

In this chapter we’re going to work our way through a gene expression analysis to demonstrate the power of SciPy to solve a real-world biological problem that I come across everyday.
Along the way we will use Pandas to parse tabular data, and then manipulate our data efficiently in Numpy ndarrays.
We will then do some analysis using packages from the R statistical language called from within Python using Rpy2.

But before we get to the juicy code, let me fill you in about my particular biological problem.
The central dogma of genetics says that all the information to run a cell is stored in the DNA.
To access this information, the DNA needs to be transcribed into messenger RNA (mRNA).
The amount of mRNA produced from a given gene is called the “expression” of that gene.
The mRNA is in turn is translated into protein, which is the workhorse of the cell.
Proteins can act as building blocks like the keratin that gives structure to your nails and hair, or the enzymes that allow you to digest your food.
Unfortunately, protein is particularly difficult to detect experimentally, but mRNA is actually pretty easy to measure.
So we make the assumption that if we measure the amount of mRNA we can gain insight into how the cells are functioning.

[insert central dogma figure?]

Currently, the most sensitive way to measure mRNA is to do an RNA sequencing (RNAseq) experiment. To do this we isolate all the mRNA from a sample, then we sequence it.
Currently, high-throughput sequencing machines can only read short fragments (approximately 100 bases is common). These short sequences are called “reads”.
We measure millions of reads and then based on their sequence we count how many reads came from each gene.
For this chapter we’ll be starting directly from this count data, but in [ch7?] we will talk more about how this type of data can be determined.

[diagram/flow chart of RNAseq]

Let’s have a look at what this gene expression data looks like.

|        | Cell type A | Cell type B |
|--------|-------------|-------------|
| Gene 1 | 100         | 200         |
| Gene 2 | 50          | 0           |
| Gene 3 | 350         | 100         |

The data is a table of counts, integers representing how many reads were observed for each gene in each cell type.
This data is perfect to represented more efficiently as a ndarray.

## Numpy N-dimensional arrays

One of the key NumPy data types is the N-dimensional array (ndarray).
Ndarrays must be homogenous; all items in an ndarray must be the same type and size.
In our case we will need to store integers.

Ndarrays are called N-dimensional because they can have any number of dimensions.
For example a 1-dimesional array would look like this:

```python
import numpy as np

one_d_array = np.array([1,2,3,4])
print(one_d_array)
```

For a 2-dimensional array, let's use our mini gene expression table from above.

```python
two_d_array = np.array([
        [100, 200],
        [50, 0],
        [350, 100]
    ])
print(two_d_array)
```

Once we get into three dimensions, things start to get trickier to imagine.

```python
three_d_array = np.array([
        [[1,2], [3,4]],
        [[5,6], [7,8]],
    ])
print(three_d_array)
```

I like to draw this as a cube.

[3-dimensional array picture]


A 4 dimensional array becomes tricky to visualise even with a diagram,
so perhaps it is easier to think of it in terms of a use case.
Let's say you have an ndarray that describes an object over time.
You would need three dimensions to describe the position of
the objects and a fourth to indicate time.

```python
# Use .shape to check the dimensions of an ndarray
print(two_d_array.shape)
print(len(two_d_array.shape))
```

```python
print(three_d_array.shape)
print(len(three_d_array.shape))
```

```python
# Use .dtype to determine the data type (including allocated memory)
print(two_d_array.dtype)
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

Ndarrays are also size efficient.
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
import pandas as pd

# Access file remotely (uncomment to use):
# url = "http://eqtl.uchicago.edu/RNA_Seq_data/results/final_gene_counts.gz"
# filehandle = urllib.request.urlopen(url)

# Access file locally:
filehandle = "../data/final_gene_counts.gz"

with gzip.open(filehandle, 'rt') as f:
    data_table = pd.read_csv(f, delim_whitespace=True)
```

Write a tiny bit about pandas
- it's useful for reading in data of mixed type then we can pull out the bits
we are interested in
- It's sort of like an R data frame
- plug the pandas book
- We'll see pandas again in later chapters

Explore the dataset

```python
skip_cols = 3

# Sample names
samples = list(data_table.columns)[skip_cols:]

# 2D ndarray containing counts for each gene in each individual
counts = np.asarray(data_table.iloc[:, skip_cols:], dtype=int)

# 1D ndarray containing the lengths of each gene
gene_lengths = np.asarray(data_table.iloc[:, 2], dtype=int)
```

```python
%matplotlib inline
# Make all plots appear inline from now onwards

import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

```python
# bar plot

small_data = counts[:10] # reduce size of data

plt.figure()
plt.boxplot(small_data)
plt.show()

```

Convert to RPKM: Reads per kilobase transcript per million reads

C = Number of reads mapped to a gene

N = Total mapped reads in the experiment

L = exon length in base-pairs for a gene

Equation = RPKM = (10^9 * C)/(N * L)


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


#### Between genes




```python
# plot of a small gene vs a big gene
# (choose two that have otherwise similar expression levels?
# or better yet big one looks like it has more expression but actually
# the little one is higher afer normalisation)
```

Convert to RPKM: Reads per kilobase transcript per million reads

Divide the number of reads by the size of the gene that they map to in kilobases
(1 kb = 1000 DNA bases) then divi

RPKM(X) = (10^9 * C) / (N * L)

C is the number of reads mapping to that gene
N is the total number of reads (sum all the counts for that individual)
L is the length of the gene in base pairs (??? not kilobasepairs?)

```python
# Convert to RPKM
# Redo plot showing new relationship between the genes
```

Between samples


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

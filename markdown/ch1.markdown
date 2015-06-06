
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
Let's say you have a list and you want to multiply every element in the list by 5.
A standard Python approach would be to write a loop that iterates through the
elements of the list and multiply each one by 5.
However, if your data were instead represented as an ndarray,
you can multiply every element in the ndarray by 5 simultaneously.

```python
    import numpy as np

    # Create an ndarray of integers in the range 0 up to (but not including) 10,000,000
    nd_array = np.arange(1e6)
    # Convert arr to a list
    list_array = nd_array.tolist()
```

```python
    %%timeit -n10 # Use the Ipython "magic" command timeit to time how long it takes to multiply each element in the ndarray by 5
    x = nd_array * 5
```

```python
    %%timeit -n10 # Time how long it takes to multiply each element in the list by 5
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

One of the most powerful and often misunderstood features of the ndarray is broadcasting.
Broadcasting is a way of performing operations between two arrays.
Earlier when we talked about the speed of operations on ndarrays, what we were actually doing was broadcasting.
Let's look at some examples.

```python
x = np.array([1, 2, 3, 4])

x * 2
```

```python
y = np.array([0, 1, 2, 1])
x + y # add every element in x to the corresponding element in y
```

```python
x * y # multiply every element in x by the corresponding element in y
```

Another word for this behavour is vectorisation, which is a key feature of array languages such as Matlab and R.
Under the hood this is eqivalent to the to a for loop, but much faster because the loop is running in C rather than Python.
Let's try the same calculation as a loop and using broadcasting to see how much of a speed up we can get.

```python
# Create an ndarray of length 10,000,000
nd_array = np.arange(1e6)
```

```python
%%timeit -n10

array_len = len(nd_array)
result = np.empty(shape=array_len, dtype="int64") # Create an empty ndarray
for i in range(array_len):
    result[i] = nd_array[i] * 5
```

```python
%%timeit -n10

result = nd_array * 5
```

## Exploring a gene expression data set

The data set that we'll be using is an RNAseq experiment of healthy individuals from a project called HapMap (http://hapmap.ncbi.nlm.nih.gov/).
This is a standard reference data set to give researchers an idea of the baseline variation between healthy individuals.
The raw sequencing reads are available at http://eqtl.uchicago.edu/RNA_Seq_data/unmapped_reads/.
We will be using this data in a later chapter.
However, in this chapter we will be starting from the gene count data, which can be found at http://eqtl.uchicago.edu/RNA_Seq_data/results/final_gene_counts.gz.
If you are curious to see a full analysis of this data set right from raw sequencing reads, the Limma R package documentaation is a great place to start (http://www.bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf).

We're first going to use Pandas to read in the table of counts.
Pandas is particularly useful for reading in tabular data of mixed type.
It uses the DataFrame type, which is a flexible tabular format based on the data frame object in R.
For example the data we will read has a column of gene names (strings) and multiple columns of counts (integers), so it doesn't make sense to read this data in directly as an ndarray.
By reading the data in as a Pandas DataFrame we can let Pandas do all the parsing, then extract out the relevant information and store it in a more efficient data type.
Here we are just using Pandas briefly to import data.
In later chapters we will give you some more insight into the world of Pandas.

```python
import urllib
import numpy as np
import pandas as pd
import os
import gzip

url = "http://eqtl.uchicago.edu/RNA_Seq_data/results/final_gene_counts.gz" # Location of remote file
filename = "final_gene_counts.gz" # Local filename

if not os.path.exists(filename): # Check if file exists
    urllib.request.urlretrieve(url, filename) # Download file

with gzip.open(filename, 'rt') as f:
    data_table = pd.read_csv(f, delim_whitespace=True) # Parse file with pandas (automatically unzips)

print(data_table.iloc[:5, :5]) # print the first 5 rows and columns of the DataFrame
```

We can see that Pandas has kindly pull out the header row and used it to name the columns.
The first three columns are information about a gene.
The ID of the gene, what chromosome it is on, and how long the gene is.
The remaining columns are IDs for the individual people who were tested, along with the name of the lab that performed the testing (Argonne National Laboratory for the first few).
Let's extract out the data that we need in a more useful format.


```python
skip_cols = 3 #

# Sample names
samples = list(data_table.columns)[skip_cols:]

# 2D ndarray containing expression counts for each gene in each individual
counts = np.asarray(data_table.iloc[:, skip_cols:], dtype=int)

# 1D ndarray containing the lengths of each gene
gene_lengths = np.asarray(data_table.iloc[:, 2], dtype=int)

# Check how many genes and individuals were measured
print("{0} genes measured in {1} individuals".format(counts.shape[0], counts.shape[1]))
```

## Differential gene expression analysis

The most common analysis done with RNAseq data is a differential gene expression analysis.
The general strategy is to compare two different groups, say disease vs. normal, treatment vs. control, and ask the question; are which genes are differentially expressed between the two groups?
In other words, which genes have significantly higher counts in one group compared with the other?

If we assume gene expression counts are normally distributed, then we could do a t-test for each gene.

```python
%matplotlib inline
# Make all plots appear inline from now onwards

import matplotlib.pyplot as plt
plt.style.use('ggplot') # Use ggplot style graphs for something a litle prettier
```

```python
import matplotlib.pyplot as plt
from scipy import stats

# Plot the density of expresison counts for a gene to check if it's approximately normal
data = counts[0, :] # expression counts for one gene
density = stats.kde.gaussian_kde(data) # Use guassian smoothing to estimate the density
x = np.arange(0, max(data)) # create ndarray of integers from 0 to largest expression count for that gene
plt.plot(x, density(x))
plt.show()
```

The expression counts are close enough to normal, so let's try the t-tests first.

The problem of what is the best way to perform a differential gene expression analysis has long been a topic of debate amoungst statisticians.
There already exists a wealth of methods, however many of them are written in R, the language of choice for many statisticians.
Lucky for us, we can use Rpy2 to run these R commands from Python.

### Normalization

Before we dive into the stats, it is important to first determine if we need to normalise our data.

#### Between samples

The number of counts for each individual can vary substantially in RNAseq experiments.
Let's take a look.

```python
total_counts = counts.sum(axis=0) # sum each column (axis=1 would sum rows)

density = stats.kde.gaussian_kde(total_counts) # Use guassian smoothing to estimate the density
x = np.arange(min(total_counts), max(total_counts), 10000) # create ndarray of integers from min to max in steps of 10,000
plt.plot(x, density(x))
plt.xlabel("Total counts per individual")
plt.show()

print("Min counts: {0}, Mean counts: {1}, Max counts: {2}".format(total_counts.min(), total_counts.mean(), total_counts.max()))
```

We can see that there is an order of magnitude difference in the total number of counts between the lowest and the highest individual.
This means that a different number of RNAseq reads were generated for each individual.
We say that these individuals have different library sizes.

```python
# Bar plot of expression counts by individual
plt.figure(figsize=(20,5))
plt.boxplot(counts, sym=".")
plt.title("Gene expression counts raw")
plt.xlabel("Individuals")
plt.ylabel("Gene expression counts")
plt.show()
```

There are obviously a lot of outliers at the high expression end of the scale and a lot of variation between individuals, but pretty hard to see because everything is clustered around zero.
So let's do log(n + 1) of our data so it's a bit easier to look at.
Both the log function and the n + 1 step can be done using broadcasting to simpify our code and speed things up.

```python
# Bar plot of expression counts by individual
plt.figure(figsize=(20,5))
plt.boxplot(np.log(counts + 1), sym=".")
plt.title("Gene expression counts raw")
plt.xlabel("Individuals")
plt.ylabel("Gene expression counts")
plt.show()
```

Now let's see what happens when we normalise by library size.

```python
# Normalise by library size
# Divide the expression counts by the total counts for that individual
counts_lib_norm = counts / total_counts * 1000000 # Multiply by 1 million to get things back in a similar scale
# Notice how we just used broadcasting twice there!

# Bar plot of expression counts by individual
plt.figure(figsize=(20,5))
plt.boxplot(np.log(counts_lib_norm + 1), sym=".")
plt.title("Gene expression counts normalised by library size")
plt.xlabel("Individuals")
plt.ylabel("Gene expression counts")
plt.show()
```

Much better!
Also notice how we used broadcasting twice there.
Once to divide all the gene expression counts by the total for that column, and then again to multiply all the values by 1 million.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it

def class_boxplot(data, classes, colors=None, **kwargs):
    """Make a boxplot with boxes colored according to the class they belong to.

    Parameters
    ----------
    data : list of array-like of float
        The input data. One boxplot will be generated for each element
        in `data`.
    classes : list of string, same length as `data`
        The class each distribution in `data` belongs to.
    colors : list of matplotlib colorspecs
        The color corresponding to each class. These will be cycled in
        the order in which the classes appear in `classes`. (So it is
        ideal to provide as many colors as there are classes! The
        default palette contains five colors.)

    Other parameters
    ----------------
    kwargs : dict
        Keyword arguments to pass on to `plt.boxplot`.
    """
    # default color palette
    if colors is None:
        colors = sns.xkcd_palette(["windows blue", "amber", "greyish",
                                   "faded green", "dusty purple"])
    # default boxplot parameters; only updated if not specified
    kwargs['sym'] = kwargs.get('sym', '.')
    kwargs['whiskerprops'] = kwargs.get('whiskerprops', {'linestyle': '-'})

    all_classes = sorted(set(classes))
    class2color = dict(zip(all_classes, it.cycle(colors)))
    # create a dictionary containing data of same length but only data
    # from that class
    class2data = {}
    for i, (distrib, cls) in enumerate(zip(data, classes)):
        for c in all_classes:
            class2data.setdefault(c, []).append([])  # empty dataset at first
        class2data[cls][-1] = distrib
    # then, do each boxplot in turn with the appropriate color
    lines = []
    for cls in all_classes:
        # set color for all elements of the boxplot
        for key in ['boxprops', 'whiskerprops', 'capprops',
                    'medianprops', 'flierprops']:
            kwargs.setdefault(key, {}).update(color=class2color[cls])
        # draw the boxplot
        box = plt.boxplot(class2data[cls], **kwargs)
        lines.append(box['caps'][0])
    plt.legend(lines, all_classes)
```

Now we can plot a colored boxplot according to normalized vs unnormalized samples.
We show only three samples from each class for illustration:

```python
class_boxplot(list(counts.T[:3]) + list(counts_lib_norm.T[:3]),
              ['raw counts'] * 3 + ['normalized by library size'] * 3)
```

An example of the types of plots I'd like to show:
http://www.nature.com/nbt/journal/v32/n9/images_article/nbt.2931-F2.jpg

#### Between genes

Number of reads related to length of gene

```python
counts = counts_lib_norm # Use normalised counts

# Bar plot log(n + 1) of expression counts by gene for the first few genes
small_data = np.log(counts + 1)[:50, :] # [rows, columns] where rows are genes and columns are individuals
small_data = small_data.transpose() # Transpose so that genes are now columns

plt.figure(figsize=(20,5))
plt.boxplot(small_data, sym=".")
plt.xlabel("Genes")
plt.ylabel("Expression counts")
#plt.plot(gene_lengths[:50] / 1000) # Also plot corresponding gene lengths (divided by 1000 to get them into the same scale)
plt.show()
```

```python
mean_counts = counts.mean(axis=1) # mean expression counts per gene
plt.figure()
plt.scatter(gene_lengths, mean_counts)
plt.xlabel("Gene length in base pairs")
plt.ylabel("Mean expression counts for that gene")
#plt.xlim(0,40000)
#plt.ylim(0,10000)
plt.show()
```

```python
# Bin the counts by gene length and then produce boxplot for each bin

def assign_bins(array, bins):
    for value in array:
        idx = (np.abs(bins-value)).argmin()
        yield bins[idx]

#Return evenly spaced numbers over a specified interval.
bins = np.linspace(10, 50000, 10)

print([i for i in assign_bins(bins, bins)])

gene_bins = [i for i in assign_bins(gene_lengths, bins)]

print(gene_lengths[:10])
#print(bins)
print(gene_bins[:10])
#print(bin_means)
#stats.binned_statistic(x, values, statistic='mean', bins=10, range=None)

# I want something like this except box plots
plt.figure()
plt.scatter(gene_bins, mean_counts)
plt.xlabel("Gene length in base pairs")
plt.ylabel("Mean expression counts for that gene")
plt.show()
```


(some simple descriptive statistics and plots PCA/MDS?)

Convert to RPKM: Reads per kilobase transcript per million reads

C = Number of reads mapped to a gene

N = Total mapped reads in the experiment

L = exon length in base-pairs for a gene

Equation = RPKM = (10^9 * C)/(N * L)


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

```python
import rpy2.robjects as robjects
```

Diagnostic plots

- P-value histogram
- Volcano plot of results


## Other topics that could be covered

### NumPy/SciPy functions to cover
np.log2
np.mean
np.sort
np.round

```python
from scipy import stats
stats.rankdata
```

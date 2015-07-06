# Elegant NumPy: The foundation of scientific computing in Python

This chapter touches on some statistical functions in SciPy, but more than that, it focuses on exploring the NumPy array, a data structure that underlies almost all numerical scientific computation in Python.
We will see how NumPy array operations enable concise and efficient code when manipulating numerical data.

Our use case is the analysis of gene expression data to predict mortality in skin cancer patients, reproducing a simplified version of [Figures 5A and 5B](http://www.cell.com/action/showImagesData?pii=S0092-8674%2815%2900634-0) of a [paper](http://dx.doi.org/10.1016/j.cell.2015.05.044) from The Cancer Genome Atlas (TCGA) project.
(We will unpack what "gene expression" means in just a moment.)

The code we will work to understand is an implementation of [*quantile normalization*](https://en.wikipedia.org/wiki/Quantile_normalization), a technique that ensures measurements fit a specific distribution.
This requires a strong assumption: if the data are not distributed according to a bell curve, we just make it fit!
But it turns out to be simple and useful in many cases where the specific distribution doesn't matter, but the relative changes of values within a population are important.
For example, Bolstad and colleagues [showed](http://bioinformatics.oxfordjournals.org/content/19/2/185.full.pdf) that it performs admirably in recovering known expression levels in microarray data.

Using NumPy indexing tricks and the `scipy.stats.rank_data` function, quantile normalization in Python is fast, efficient, and elegant.

```python
import numpy as np
from scipy import stats

def quantile_norm(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of ngenes
    by nsamples, quantile normalization ensures all samples have the same
    spread of data (by construction).

    The input data is log-transformed. The rows are averaged and each column
    quantile is replaced with the quantile of the average column.
    The data is then transformed back to counts.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with n_features rows and n_samples columns.

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # log-transform the data
    logX = np.log2(X + 1)

    # compute the quantiles
    log_quantiles = np.mean(np.sort(logX, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 0, the
    # second-smallest by 1, ..., and the largest by M, the number of rows.
    ranks = np.transpose([np.round(stats.rankdata(col)).astype(int) - 1
                          for col in X.T])

    # index the quantiles for each rank with the ranks matrix
    logXn = log_quantiles[ranks]

    # convert the data back to counts (casting to int is optional)
    Xn = np.round(2**logXn - 1).astype(int)
    return(Xn)
```

We'll unpack that example throughout the chapter, but for now note that it illustrates many of the things that make NumPy powerful:

- Arrays can be one-dimensional, like lists, but they can also be two-dimensional, like matrices, and higher-dimensional still. This allows them to represent many different kinds of numerical data. In our case, we are representing a 2D matrix.
- Arrays allow the expression of many numerical operations at once. In the first line of the function, we take $log(x + 1)$ for every value in the array.
- Arrays can be operated on along *axes*. In the second line, we sort the data along each column just by specifying an `axis` parameter to `np.sort`. We then take the mean along each row by specifying a *different* `axis`.
- Arrays underpin the scientific Python ecosystem. The `scipy.stats.rankdata` function operates not on Python lists, but on NumPy arrays. This is true of many scientific libraries in Python.
- Arrays support many kinds of data manipulation through *fancy indexing*: `logXn = log_quantiles[ranks]`. This is possibly the trickiest part of NumPy, but also the most useful. We will explore it further in the text that follows.

Before we delve into the power of NumPy, let's spend some time to understand the biological data that we will be working with.

## What is gene expression?

We will work our way through a *gene expression analysis* to demonstrate the power of NumPy and SciPy to solve a real-world biological problem.
We will use the Pandas library, which builds on NumPy, to read and munge our data files, and then we manipulate our data efficiently in NumPy arrays.

The central dogma of molecular biology states that all the information needed to run a cell (or an organism, for that matter) is stored in a molecule called *deoxyribonucleic acid*, or DNA.
This molecule has a repetitive backbone on which lie chemical groups called *bases*, in sequence.
There are four kinds of bases, abbreviated to A, C, G, and T, constituting an alphabet with which information is stored.

![The chemical structure of DNA](https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg)
*Image by Madeleine Price Ball, used under the terms of the CC0 public domain license*

To access this information, the DNA is *transcribed* into a sister molecule called *messenger ribonucleic acid*, or mRNA.
Finally, this mRNA is *translated* into proteins, the workhorses of the cell.

The amount of mRNA produced from a given gene is called the *expression* of that gene.
Although we would ideally like to measure protein levels, this is a much harder task than measuring mRNA.
Fortunately, expression levels of an mRNA and levels of its corresponding protein are usually correlated ([Maier, Güell, and Serrano, 2009](http://www.sciencedirect.com/science/article/pii/S0014579309008126)).
Therefore, we usually measure mRNA levels and base our analyses on that.
As you will see below, it often doesn't matter, because we are using mRNA levels for their power to predict biological outcomes, rather than to make specific statements about proteins.

![Central Dogma of Biology](http://www.phschool.com/science/biology_place/biocoach/images/transcription/centdog.gif)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

It's important to note that the DNA in every cell of your body is identical.
Thus, the differences between cells arise from *differential expression* of that DNA into RNA.
Similarly, as we shall see in this chapter, differential expression can distinguish different kinds of cancer.

![Gene expression](http://www.ncbi.nlm.nih.gov/Class/MLACourse/Original8Hour/Genetics/cgap_conceptual_tour1.gif)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

The state-of-the-art technology to measure mRNA is RNA sequencing (RNAseq).
RNA is extracted from a tissue, for example from a biopsy from a patient, *reverse transcribed* back into DNA (which is more stable), and then read out using chemically modified bases that glow when they are incorporated into the DNA sequence.
Currently, high-throughput sequencing machines can only read short fragments (approximately 100 bases is common). These short sequences are called “reads”.
We measure millions of reads and then based on their sequence we count how many reads came from each gene.
For this chapter we’ll be starting directly from this count data, but in [ch7?] we will talk more about how this type of data can be determined.

![RNAseq](http://bio.lundberg.gu.se/courses/vt13/rna4.JPG.jpg)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

Here's an example of what this gene expression data looks like.

|        | Cell type A | Cell type B |
|--------|-------------|-------------|
| Gene 1 | 100         | 200         |
| Gene 2 | 50          | 0           |
| Gene 3 | 350         | 100         |

The data is a table of counts, integers representing how many reads were observed for each gene in each cell type.
See how the counts for each gene differ between the cell types?
We can use this information to tell us about the differences between these two types of cell.
This data is perfect to represented more efficiently as a ndarray.

## NumPy N-dimensional arrays

One of the key NumPy data types is the N-dimensional array (ndarray, or just array).
Arrays must be homogeneous; all items in an array must be the same type.
In our case we will need to store integers.

Ndarrays are called N-dimensional because they can have any number of dimensions.
For example a 1-dimesional array would look like this:

```python
import numpy as np

one_d_array = np.array([1,2,3,4])
print(one_d_array)
```

```python
# Use .shape to check the dimensions of an ndarray
print(one_d_array.shape)
print(len(one_d_array.shape))
```

Remember that arrays must contain all elements of the same type.
The type is set automatically from the input data used to create the array.
NumPy will choose the minimum type required to hold all the objects.
The type can also be set explicitly.

```python
# Use .dtype to determine the data type (including allocated memory)
print(one_d_array)
print(one_d_array.dtype)
one_d_array = np.array([1,2,3,4], dtype='str') # Set the data type to string (upcasting)
print(one_d_array)
print(one_d_array.dtype)
```

For a 2-dimensional array, let's use our mini gene expression table from above.

```python
two_d_array = np.array([[100, 200],
                        [ 50,   0],
                        [350, 100]])
print(two_d_array)
```

![2-dimensional array diagram](http://www.inf.ethz.ch/personal/gonnet/DarwinManual/img24.gif)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

```python
# Use .shape to check the dimensions of an ndarray
print(two_d_array.shape)
print(len(two_d_array.shape))
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

![3-dimensional array diagram](http://www.inf.ethz.ch/personal/gonnet/DarwinManual/img25.gif)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

```python
print(three_d_array.shape)
print(len(three_d_array.shape))
```

A 4 dimensional array becomes tricky to visualize even with a diagram,
so perhaps it is easier to think of it in terms of a use case.
Let's say you have an ndarray that describes an object over time.
You would need three dimensions to describe the position of
the objects and a fourth to indicate time.

### Why use ndarrays as opposed to Python lists?

Arrays are fast because they enable vectorized operations, written in the low-level language C, that act on the whole array.
Say you have a list and you want to multiply every element in the list by 5.
A standard Python approach would be to write a loop that iterates over the
elements of the list and multiply each one by 5.
However, if your data were instead represented as an array,
you can multiply every element in the array by 5 in a single bound.
Behind the scenes, the highly-optimized NumPy library is doing the iteration as fast as possible.

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
(or in this case the `arange` function)
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
    # If you actually wanted a copy of your array use the copy function
    y = np.copy(x[:2])
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

Another word for this behavior is vectorization, which is a key feature of array languages such as Matlab and R.
Under the hood this is equivalent to the to a for loop, but much faster because the loop is running in C rather than Python.
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

We will come back to some more advanced broadcasting examples as we start to deal with real data.

## Exploring a gene expression data set

The data set that we'll be using is an RNAseq experiment of skin cancer samples from The Cancer Genome Atlas (TCGA) project (http://cancergenome.nih.gov/).
We will be using this gene expression data to predict mortality in skin cancer patients, reproducing a simplified version of [Figures 5A and 5B](http://www.cell.com/action/showImagesData?pii=S0092-8674%2815%2900634-0) of a [paper](http://dx.doi.org/10.1016/j.cell.2015.05.044).

### Downloading the data

[Links to data!]

We're first going to use Pandas to read in the table of counts.
Pandas is particularly useful for reading in tabular data of mixed type.
It uses the DataFrame type, which is a flexible tabular format based on the data frame object in R.
For example the data we will read has a column of gene names (strings) and multiple columns of counts (integers), so it doesn't make sense to read this data in directly as an ndarray.
By reading the data in as a Pandas DataFrame we can let Pandas do all the parsing, then extract out the relevant information and store it in a more efficient data type.
Here we are just using Pandas briefly to import data.
In later chapters we will give you some more insight into the world of Pandas.

```python
import numpy as np
import pandas as pd

# Import TCGA melanoma data
filename = 'data/counts.txt'
with open(filename, 'rt') as f:
    data_table = pd.read_csv(f, index_col=0) # Parse file with pandas

print(data_table.iloc[:5, :5])
```

We can see that Pandas has kindly pull out the header row and used it to name the columns.
The first column gives the name of the gene, and the remaining columns represent individual samples.

We will also needs some corresponding metadata,
including the sample information and the gene lengths.

```python
# Sample names
samples = list(data_table.columns)
```

```python
# Import gene lengths
filename = 'data/genes.csv'
with open(filename, 'rt') as f:
    gene_info = pd.read_csv(f, index_col=0) # Parse file with pandas, index by GeneSymbol
print(gene_info.iloc[:5, :5])
```

```python
#Subset gene info to match the count data
matched_index = data_table.index.intersection(gene_info.index)
print(gene_info.loc[matched_index].shape)
print(data_table.loc[matched_index].shape)
```

```python
# 1D ndarray containing the lengths of each gene
gene_lengths = np.asarray(gene_info.loc[matched_index]['GeneLength'],
                          dtype=int)
```

```python
# 2D ndarray containing expression counts for each gene in each individual
counts = np.asarray(data_table.loc[matched_index], dtype=int)

# Check how many genes and individuals were measured
print("{0} genes measured in {1} individuals".format(counts.shape[0], counts.shape[1]))
```

## Normalization

Before we dive into the stats, it is important to first determine if we need to normalize our data.

### Between samples

For example, the number of counts for each individual can vary substantially in RNAseq experiments.
Let's take a look.

```python
%matplotlib inline
# Make all plots appear inline in the IPython notebook from now onwards

import matplotlib.pyplot as plt
plt.style.use('ggplot') # Use ggplot style graphs for something a little prettier
```

```python
total_counts = counts.sum(axis=0) # sum each column (axis=1 would sum rows)

density = stats.kde.gaussian_kde(total_counts) # Use gaussian smoothing to estimate the density
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
# Subset data for plotting
np.random.seed(seed=7) # Set seed so we will get consistent results
samples_index = np.random.choice(range(counts.shape[1]), size=70, replace=False) # Randomly select 70 samples
counts_subset = counts[:,samples_index]
```

```python
# Bar plot of expression counts by individual
plt.figure(figsize=(16,5))
plt.boxplot(counts_subset, sym=".")
plt.title("Gene expression counts raw")
plt.xlabel("Individuals")
plt.ylabel("Gene expression counts")
plt.show()
```

There are obviously a lot of outliers at the high expression end of the scale and a lot of variation between individuals, but pretty hard to see because everything is clustered around zero.
So let's do log(n + 1) of our data so it's a bit easier to look at.
Both the log function and the n + 1 step can be done using broadcasting to simplify our code and speed things up.

```python
# Bar plot of expression counts by individual
plt.figure(figsize=(16,5))
plt.boxplot(np.log(counts_subset + 1), sym=".")
plt.title("Gene expression counts raw")
plt.xlabel("Individuals")
plt.ylabel("Gene expression counts")
plt.show()
```

Now let's see what happens when we normalize by library size.

```python
# normalize by library size
# Divide the expression counts by the total counts for that individual
counts_lib_norm = counts / total_counts * 1000000 # Multiply by 1 million to get things back in a similar scale
# Notice how we just used broadcasting twice there!
counts_subset_lib_norm = counts_lib_norm[:,samples_index]

# Bar plot of expression counts by individual
plt.figure(figsize=(16,5))
plt.boxplot(np.log(counts_subset_lib_norm + 1), sym=".")
plt.title("Gene expression counts normalized by library size")
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
log_counts_3 = list(np.log(counts.T[:3] + 1))
log_ncounts_3 = list(np.log(counts_lib_norm.T[:3] + 1))
class_boxplot(log_counts_3 + log_ncounts_3,
              ['raw counts'] * 3 + ['normalized by library size'] * 3,
              labels=[1, 2, 3, 1, 2, 3])
plt.xlabel('sample number')
plt.ylabel('log gene expression counts')
plt.show()
```

### Between genes

The number of counts for a gene, is related to the gene length.
Let's say we have gene A and gene B.
Gene B is twice as long as gene A.
Both are expressed at similar levels in the sample, i.e. both produce a similar number of mRNA molecules.
Therefore you would expect that gene B would have about twice as many counts as gene A.
Remember, that when we do an RNAseq experiement, we are fragmenting the transcript, and sampling reads from that pool of fragments.
The counts are the number of reads from that gene in a given sample.
So if a gene is twice as long, we are twice as likly to sample it.

![Relationship between counts and gene length](https://izabelcavassim.files.wordpress.com/2015/03/screenshot-from-2015-03-08-2245511.png)
**[ED NOTE, this is a placeholder image only. We do not have license to use it.]**

Let's see if the relationship between gene length and counts plays out in our data set.

```python
def binned_boxplot(x, y):
    """
    x: x axis, values to be binned. Should be a 1D ndarray.
    y: y axis. Should be a 1D ndarray.
    """
    # get the "optimal" bin size using astropy's histogram function
    from astropy.stats import histogram
    gene_len_hist, gene_len_bins = histogram(log_gene_lengths, bins='knuth')
    # np.digitize tells you which bin an observation belongs to.
    # we don't use the last bin edge because it breaks the right-open assumption
    # of digitize. The max observation correctly goes into the last bin.
    gene_len_idxs = np.digitize(log_gene_lengths, gene_len_bins[:-1])
    # Use those indices to create a list of arrays, each containing the log
    # counts corresponding to genes of that length. This is the input expected
    # by plt.boxplot
    binned_counts = [mean_log_counts[gene_len_idxs == i]
                     for i in range(np.max(gene_len_idxs))]
    plt.figure(figsize=(16,3))
    # Make the x-axis labels using real gene length
    gene_len_bin_centres = (gene_len_bins[1:] + gene_len_bins[:-1]) / 2
    gene_len_labels = np.round(np.exp(gene_len_bin_centres)).astype(int)
    # use only every 5th label to prevent crowding on x-axis ticks
    labels = []
    for i, lab in enumerate(gene_len_labels):
        if i % 5 == 0:
            labels.append(str(lab))
        else:
            labels.append('')
    # make the boxplot
    plt.boxplot(binned_counts, labels=labels, sym=".")
    # Adjust the axis names
    plt.xlabel('gene length (log scale)')
    plt.ylabel('average log-counts')
    plt.show()

log_counts = np.log(counts_lib_norm + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)

binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
```

We can see a positive relationship between the length of a gene and the counts!

### Normalizing over samples and genes: RPKM

One of the simplest normalization methods for RNAseq data is RPKM: reads per
kilobase transcript per million reads.
RPKM puts together the ideas of normalising by sample and by gene.
When we calculate RPKM, we are normalizing for both the library size (the sum of each column)
and the gene length.

Working through how RPKM is derived:

Let's say:  
C = Number of reads mapped to a gene  
L = exon length in base-pairs for a gene  
N = Total mapped reads in the experiment  

First, let's calculate reads per kilobase.

Reads per base would be:  
$\frac{C}{L}$

The formula asks for reads per kilobase instead of reads per base.
One kilobase = 1000 bases, so we'll need to divide length (L) by 1000.

Reads per kilobase would be:  
$\frac{C}{L/1000}  = \frac{10^3C}{L}$

Next, we need to normalize by library size.
If we just divide by the number of mapped reads we get:  
$ \frac{10^3C}{LN} $

But biologists like thinking in millions of reads so that the numbers don't get
too big. Counting per million reads we get:  
$ \frac{10^3C}{L(N/10^6)} = \frac{10^9C}{LN}$


In summary, to calculate reads per kilobase transcript per million reads:  
$RPKM = \frac{10^9C}{LN}$

Where:  
C = Number of reads mapped to a gene  
N = Total mapped reads in the experiment  
L = exon length in base-pairs for a gene

Now let's implement RPKM over the entire counts array.

```python
# Make our variable names the same as the RPKM formula so we can compare easily
C = counts
N = counts.sum(axis=0) # sum each column to get total reads per sample
L = gene_lengths # lengths for each gene in the same order as the rows in counts
```

First, we multiply by 10^9.
Because counts (C) is an ndarray, we can use broadcasting.
If we multiple an ndarray by a single value,
that value is broadcast over the entire array.

```python
# Multiply all counts by 10^9
C_tmp = 10^9 * C
```
Next we need to divide by the gene length.
Broadcasting a single value over a 2D array was pretty clear.
We were just multiplying every element in the array by the value.
But what happens when we need to divide a 2D array by a 1D array?

#### Broadcasting rules [tip box?]

Broadcasting allows calculations between ndarrays that have a different
number of dimensions.

If the input arrays do not have the same number of dimensions,
then then an additional dimension is added to the start of the first array,
with a value of 1.
Once the two arrays have the same number of dimensions,
broadcasting can only occur if the sizes of the dimensions match,
or one of them is equal to 1.

For example, let's say we have two ndarrays, A and B:  
A.shape = (1, 2)  
B.shape = (2,)

If we performed the operation `A * B` then broadcasting would occur.
B has fewer dimension than A, so during the calculation
a new dimension is prepended to B with value 1.  
B.shape = (1, 2)  
Now A and B have the same number of dimension, so broadcasting can proceed.

Now let's say we have another ndarray, C:  
C.shape = (2, 1)  
B.shape = (2,)  
Now, if we were to do the operation `C * B`,
a new dimension needs to be prepended to B.  
B.shape = (1, 2)  
However, the dimensions of the two ndarrays do not match,
so broadcasting will fail.

Let's say that we know that it is appropriate to broadcast B over C.
We can explicitly add a new dimension to B using `np.newaxis`.
Let's see this in our normalization by RPKM.

Let's have a look at the dimensions of our two arrays.

```python
# Check the shapes of C_tmp and L
print('C_tmp.shape', C_tmp.shape)
print('L.shape', L.shape)
```

We can see that C_tmp has 2 dimensions, while L has one.
So during broadcasting, an additional dimension will be prepended to L.
Then we will have:  
C_tmp.shape (20500, 375)  
L.shape (1, 20500)

The dimensions won't match!
We want to broadcast L over the first dimension of C_temp,
so we need to adjust the dimensions of L ourselves.

```python
L = L[:, np.newaxis] # append a dimension to L, with value 1
print('C_tmp.shape', C_tmp.shape)
print('L.shape', L.shape)
```

Now that our dimensions match or are equal to 1, we can broadcast.

```python
# Divide each row by the gene length for that gene (L)
C_tmp = C_tmp / L
```

Finally we need to normalize by the libaray size,
the total number of counts for that column.
Remember that we have already calculated N.

`N = counts.sum(axis=0) # sum each column to get total reads per sample`

```python
# Check the shapes of C_tmp and N
print('C_tmp.shape', C_tmp.shape)
print('N.shape', N.shape)
```

Once we trigger broadcasting, an additional dimension will be prepended to N:  
N.shape (1, 375)  
The dimensions will match so we don't have to do anything.
However, for readability, it can be useful to add the extra dimension to N anyway.

```python
# Divide each column by the total counts for that column (N)
N = N[np.newaxis, :]
print('C_tmp.shape', C_tmp.shape)
print('N.shape', N.shape)
```

```python
# Divide each column by the total counts for that column (N)
rpkm_counts = C_tmp / N
```

Let's put this in a function so we can reuse it.

```python
def rpkm(data, lengths):
    """calculate reads per kilobase transcript per million reads
    RPKM = (10^9 * C) / (N * L)

    Where:  
    C = Number of reads mapped to a gene  
    N = Total mapped reads in the experiment  
    L = exon length in base-pairs for a gene

    data: 2d ndarray of counts where columns are individual samples and rows
        are genes
    lengths: list or 1d nd array of the gene lengths in bp in the same order
        as the rows
    """

    N = data.sum(axis=0) # sum each column to get total reads per sample
    L = lengths
    C = data

    rpkm = ( (10^9 * C) / N[np.newaxis, :] ) / L[:, np.newaxis]

    return(rpkm)

counts_rpkm = rpkm(counts, gene_lengths)  
```

```python
# Repeat binned boxplot with raw values
log_counts = np.log(counts + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)

binned_boxplot(x=log_gene_lengths, y=mean_log_counts)

# Repeat binned boxplot with RPKM values
log_counts = np.log(counts_rpkm + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)

binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
```

RPMK normalization can be particularly useful comparing the expression profile of two different genes.
We've already seen that longer genes have higher counts, but this doesn't mean their expression level is actually higher.
Let's choose a short gene and a long gene and compare their counts before and after RPKM normalization to see what we mean.

```python
# Boxplot of expression from a short gene vs. a long gene
# showing how normalization can influence interpretation

genes2_idx = [108, 103]
genes2_lengths = gene_lengths[genes2_idx]
genes2_labels = ['Gene A, {}bp'.format(genes2_lengths[0]), 'Gene B, {}bp'.format(genes2_lengths[1])]

log_counts_2 = list(np.log(counts[genes2_idx] + 1))
log_ncounts_2 = list(np.log(counts_rpkm[genes2_idx] + 1))

class_boxplot(log_counts_2,
              ['raw counts'] * 3,
              labels=genes2_labels)
plt.xlabel('Genes')
plt.ylabel('log gene expression counts over all samples')
plt.show()

class_boxplot(log_ncounts_2,
              ['RPKM normalized'] * 3,
              labels=genes2_labels)
plt.xlabel('Genes')
plt.ylabel('log RPKM gene expression counts over all samples')
plt.show()
```

Just looking at the raw counts, it looks like the shorter gene A is not expressed, while gene B has some gene expression.
Once we normalize to RPKM values, the story changes substantially.
Now it looks like gene A is actually expressed at a higher level than gene B.
This is because RPKM includes normalization for gene length, so we can now directly compare between genes of dramatically different lengths.

## Quantile normalization with NumPy and SciPy

```python
def plot_col_density(data, xlabel=None):
    """For each column (individual) produce a density plot over all rows (genes).

    data : 2d nparray
    xlabel : x axis label
    """

    density_per_col = [stats.kde.gaussian_kde(col) for col in data.T] # Use gaussian smoothing to estimate the density
    x = np.linspace(np.min(data), np.max(data), 100)

    plt.figure()
    for density in density_per_col:
        plt.plot(x, density(x))
    plt.xlabel(xlabel)
    plt.show()


# Before normalization
log_counts = np.log(counts + 1)
plot_col_density(log_counts, xlabel="Log count distribution for each individual")
```

Given an expression matrix (microarray data, read counts, etc) of ngenes by nsamples, quantile normalization ensures all samples have the same spread of data (by construction). It involves:

(optionally) log-transforming the data
sorting all the data points column-wise
averaging the rows
replacing each column quantile with the quantile of the average column.
This can be done with NumPy and scipy easily and efficiently.
Assume we've read in the input matrix as X:

```python
import numpy as np
from scipy import stats

def quantile_norm(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of ngenes
    by nsamples, quantile normalization ensures all samples have the same spread of data (by construction).

    The input data is log-transformed. The rows are averaged and each column
    quantile is replaced with the quantile of the average column.
    The data is then transformed back to counts.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with n_features rows and n_samples columns.

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # log-transform the data
    logX = np.log2(X + 1)

    # compute the quantiles
    log_quantiles = np.mean(np.sort(logX, axis=0), axis=1)

    # compute the column-wise ranks; need to do a round-trip through list
    ranks = np.transpose([np.round(stats.rankdata(col)).astype(int) - 1
                        for col in X.T])
    # alternative: ranks = np.argsort(np.argsort(X, axis=0), axis=0)

    # index the quantiles for each rank with the ranks matrix
    logXn = log_quantiles[ranks]

    # convert the data back to counts (casting to int is optional)
    Xn = np.round(2**logXn - 1).astype(int)
    return(Xn)

# Example usage: quantile_norm(counts_lib_norm)
```

```python
# After normalization
log_quant_norm_counts = np.log(quantile_norm(counts)+1)

plot_col_density(log_quant_norm_counts, xlabel="Log count distribution for each individual")
```


## Principal Components Analysis

```python
from sklearn.decomposition import PCA

def PCA_plot(data):
    """Plot the first two principle components of the data

    Parameters
    ----------
    data : 2D ndarray of counts (assumed to be genes X samples)
    """

    #construct your NumPy array of data
    counts_transposed = np.array(data).T

    # Set up PCA, set number of components
    pca = PCA(n_components=2)

    # project data into PCA space
    pca_transformed = pca.fit_transform(counts_transposed)

    # Plot the first two principal components
    plt.scatter(x=pca_transformed[:,0], y=pca_transformed[:,1])
    plt.title("PCA")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()
```

```python
PCA_plot(counts)
PCA_plot(quantile_norm(counts))
```

## Biclustering the counts data

Now that the data are normalized, we can cluster the genes (rows) and samples (columns) of the expression matrix.
Clustering the rows tells us which genes' expression values are linked, which is an indication that they work together in the process being studied.
Clustering the samples tells us which samples have similar gene expression profiles, which may indicate similar characteristics of the samples on other scales.

Because clustering can be an expensive operation, we will limit our analysis to the 1,500 genes that are most variable, since these will account for most of the correlation signal in either dimension.

```python
def most_variable_rows(data, n=1500):
    """Subset data to the n most variable rows

    In this case, we want the n most variable genes.

    Parameters
    ----------
    data : 2D array of float
        The data to be subset
    n : int, optional
        Number of rows to return.
    method : function, optional
        The function with which to compute variance. Must take an array
        of shape (nrows, ncols) and an axis parameter and return an
        array of shape (nrows,).
    """
    # compute variance along the columns axis
    rowvar = np.var(data, axis=1)
    # Get sorted indices (ascending order), take the last n
    sort_indices = np.argsort(rowvar)[-n:]
    # use as index for data
    variable_data = data[sort_indices, :]
    return variable_data
```

Next, we need a function to *bicluster* the data.
This means clustering along both the rows (to find out with genes are working together) and the columns (to find out which samples are similar).

Normally, you would use a sophisticated clustering algorithm from the [scikit-learn](http://scikit-learn.org) library for this.
In our case, we want to use hierarchical clustering for simplicity and ease of display.
The SciPy library happens to have a perfectly good hierarchical clustering module, though it requires a bit of wrangling to get your head around its interface.

As a reminder, hierarchical clustering is a method to group observations using sequential merging of clusters:
initially, every observation is its own cluster.
Then, the two nearest clusters are repeatedly merged, until every observation is in a single cluster.
This sequence of merges forms a *merge tree*.
By cutting the tree at a particular distance threshold, we can get a finer or coarser clustering of observations.

The `linkage` function in `scipy.cluster.hierarchy` performs a hierarchical clustering of the rows of a matrix, using a particular metric (for example, Euclidean distance, Manhattan distance, or others) and a particular linkage method, the distance between two clusters (for example, the average distance between all the observations in a pair of clusters).

It returns the merge tree as a "linkage matrix", which contains each merge operation along with the distance computed for the merge and the number of observations in the resulting cluster. From the `linkage` documentation:

> A cluster with an index less than $n$ corresponds to one of
> the $n$ original observations. The distance between
> clusters `Z[i, 0]` and `Z[i, 1]` is given by `Z[i, 2]`. The
> fourth value `Z[i, 3]` represents the number of original
> observations in the newly formed cluster.

Whew! So that's a lot of information, but let's dive right in and hopefully you'll get the hang of it rather quickly.
First, we define a function, `bicluster`, that clusters both the rows *and* the columns of a matrix:

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list


def bicluster(data, linkage_method='average', distance_metric='correlation'):
    """Cluster the rows and the columns of a matrix.

    Parameters
    ----------
    data : 2D ndarray
        The input data to bicluster.
    linkage_method : string, optional
        Method to be passed to `linkage`.
    distance_metric : string, optional
        Distance metric to use for clustering. See the documentation
        for ``scipy.spatial.distance.pdist`` for valid metrics.

    Returns
    -------
    y_rows : linkage matrix
        The clustering of the rows of the input data.
    y_cols : linkage matrix
        The clustering of the cols of the input data.
    """
    y_rows = linkage(data, method=linkage_method, metric=distance_metric)
    y_cols = linkage(data.T, method=linkage_method, metric=distance_metric)
    return y_rows, y_cols
```

Simple: we just call `linkage` for the input matrix and also for the transpose of that matrix, in which columns become rows and rows become columns.

Next, we define a function to visualize the output of that clustering.
We are going to rearrange the rows an columns of the input data so that similar rows are together and similar columns are together.
And we are additionally going to show the merge tree for both rows and columns, displaying which observations belong together for each.

As a word of warning, there is a fair bit of hard-coding of parameters going on here.
This is difficult to avoid for plotting, where design is often a matter of eyeballing to find the correct proportions.

```python
def plot_bicluster(data, row_linkage, col_linkage,
                   row_nclusters=10, col_nclusters=3):
    """Perform a biclustering, plot a heatmap with dendrograms on each axis.

    Parameters
    ----------
    data : array of float, shape (M, N)
        The input data to bicluster.
    row_linkage : array, shape (M-1, 4)
        The linkage matrix for the rows of `data`.
    col_linkage : array, shape (N-1, 4)
        The linkage matrix for the columns of `data`.
    n_clusters_r, n_clusters_c : int, optional
        Number of clusters for rows and columns.
    """
    fig = plt.figure(figsize=(8, 8))

    # Compute and plot row-wise dendrogram
    # `add_axes` takes a "rectangle" input to add a subplot to a figure.
    # The figure is considered to have side-length 1 on each side, and its
    # bottom-left corner is at (0, 0).
    # The measurements passed to `add_axes` are the left, bottom, width, and
    # height of the subplot. Thus, to draw the left dendrogram (for the rows),
    # we create a rectangle whose bottom-left corner is at (0.09, 0.1), and
    # measuring 0.2 in width and 0.6 in height.
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # For a given number of clusters, we can obtain a cut of the linkage
    # tree by looking at the corresponding distance annotation in the linkage
    # matrix.
    threshold_r = (row_linkage[-row_nclusters, 2] +
                   row_linkage[-row_nclusters+1, 2]) / 2
    dendrogram(row_linkage, orientation='right', color_threshold=threshold_r)

    # Compute and plot column-wise dendogram
    # See notes above for explanation of parameters to `add_axes`
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    threshold_c = (col_linkage[-col_nclusters, 2] +
                   col_linkage[-col_nclusters+1, 2]) / 2
    dendrogram(col_linkage, color_threshold=threshold_c)

    # Hide axes labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot data heatmap
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    # Sort data by the dendogram leaves
    idx_rows = leaves_list(row_linkage)
    data = data[idx_rows, :]
    idx_cols = leaves_list(col_linkage)
    data = data[:, idx_cols]

    im = ax.matshow(data, aspect='auto', origin='lower', cmap='YlGnBu_r')
    ax.set_xticks([])
    ax.set_yticks([])

    # Axis labels
    plt.xlabel('Samples')
    plt.ylabel('Genes', labelpad=125)

    # Plot legend
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)

    # display the plot
    plt.show()
```

Now we apply these functions to our normalized counts matrix to display row and column clusterings.

```python
counts_log = np.log(counts + 1)
counts_var = most_variable_rows(counts_log, n=1500)
yr, yc = bicluster(counts_var)
plot_bicluster(counts_var, yr, yc)
```

We can see that the sample data naturally falls into at least 2 clusters.
Are these clusters meaningful?
To answer this, we can access the patient data, available from the [data repository](https://tcga-data.nci.nih.gov/docs/publications/skcm_2015/) for the paper.
After some preprocessing, we get the [patients table]() (LINK TO FINAL PATIENTS TABLE), which contains survival information for each patient.
We can then match these to the counts clusters, and understand whether the patients' gene expression can predict differences in their pathology.

```python
patients = pd.read_csv('data/patients.csv', index_col=0)
patients.head()
```

Now we need to draw *survival curves* for each group of patients defined by the clustering.
This is a plot of the fraction of a population that remains alive over a period of time.
Note that some data is *right-censored*, which means that in some cases, don't actually know when the patient died, or the patient might have died of causes unrelated to the melanoma.
We counts these patients as "alive" for the duration of the survival curve, but more sophisticated analyses might try to estimate their likely time of death.

To obtain a survival curve from survival times, we create a step function that decreases by $1/n$ at each step, where $n$ is the population size.
We then match that function against the non-censored survival times.

```python
def survival_distribution_function(lifetimes, right_censored=None):
    """Return the survival distribution function of a set of lifetimes.

    Parameters
    ----------
    lifetimes : array of float or int
        The observed lifetimes of a population. These must be non-
        -negative.
    right_censored : array of bool, same shape as `lifetimes`
        A value of `True` here indicates that this lifetime was not
        observed. Values of `np.nan` in `lifetimes` are also considered
        to be right-censored.

    Returns
    -------
    sorted_lifetimes : array of float
        The
    sdf : array of float
        Values starting at 1 and progressively decreasing, one level
        for each observation in `lifetimes`.

    Examples
    --------

    In this example, of a population of four, two die at time 1, a
    third dies at time 2, and a final individual dies at an unknown
    time. (Hence, ``np.nan``.)

    >>> lifetimes = np.array([2, 1, 1, np.nan])
    >>> survival_distribution_function(lifetimes)
    (array([ 0.,  1.,  1.,  2.]), array([ 1.  ,  0.75,  0.5 ,  0.25]))
    """
    n_obs = len(lifetimes)
    rc = np.isnan(lifetimes)
    if right_censored is not None:
        rc |= right_censored
    observed = lifetimes[~rc]
    xs = np.concatenate(([0], np.sort(observed)))
    ys = np.concatenate((np.arange(1, 0, -1/n_obs), [0]))
    ys = ys[:len(xs)]
    return xs, ys
```

Now that we can easily obtain survival curves from the survival data, we can plot them.
We write a function that groups the survival times by cluster identity and plots each group as a different line:

```python
def plot_cluster_survival_curves(clusters, sample_names, patients,
                                 censor=True):
    """Plot the survival data from a set of sample clusters.

    Parameters
    ----------
    clusters : array of int or categorical pd.Series
        The cluster identity of each sample, encoded as a simple int
        or as a pandas categorical variable.
    sample_names : list of string
        The name corresponding to each sample. Must be the same length
        as `clusters`.
    patients : pandas.DataFrame
        The DataFrame containing survival information for each patient.
        The indices of this DataFrame must correspond to the
        `sample_names`. Samples not represented in this list will be
        ignored.
    censor : bool, optional
        If `True`, use `patients['melanoma-dead']` to right-censor the
        survival data.
    """
    plt.figure()
    if type(clusters) == np.ndarray:
        cluster_ids = np.unique(clusters)
        cluster_names = ['cluster {}'.format(i) for i in cluster_ids]
    elif type(clusters) == pd.Series:
        cluster_ids = clusters.cat.categories
        cluster_names = list(cluster_ids)
    n_clusters = len(cluster_ids)
    for c in cluster_ids:
        clust_samples = np.flatnonzero(clusters == c)
        # discard patients not present in survival data
        clust_samples = [sample_names[i] for i in clust_samples
                         if sample_names[i] in patients.index]
        patient_cluster = patients.loc[clust_samples]
        survival_times = np.array(patient_cluster['melanoma-survival-time'])
        if censor:
            censored = ~np.array(patient_cluster['melanoma-dead']).astype(bool)
        else:
            censored = None
        stimes, sfracs = survival_distribution_function(survival_times,
                                                        censored)
        plt.plot(stimes / 365, sfracs)

    plt.xlabel('survival time (years)')
    plt.ylabel('fraction alive')
    plt.legend(cluster_names)
```

Now we can use the `fcluster` function to obtain cluster identities for the samples (columns of the counts data), and plot each survival curve separately.
The `fcluster` function takes a linkage matrix, as returned by `linkage`, and a threshold, and returns cluster identities.
It's difficult to know a-priori what the threshold should be, but we can obtain the appropriate threshold for a fixed number of clusters by checking the distances in the linkage matrix.

```python
n_clusters = 3
threshold_distance = (yc[-n_clusters, 2] + yc[-n_clusters+1, 2]) / 2
clusters = fcluster(yc, threshold_distance, 'distance')

plot_cluster_survival_curves(clusters, data_table.columns, patients)
```

The clustering of gene expression profiles has identified a higher-risk subtype of melanoma, which constitutes the majority of patients.
This is indeed only the latest study to show such a result, with others identifying subtypes of leukemia (blood cancer), gut cancer, and more.
Although the above clustering technique is quite fragile, there are other ways to explore this dataset and similar ones that are more robust.

**Exercise:** We leave you the exercise of implementing the approach described in the paper:

1. Take bootstrap samples (random choice with replacement) of the genes used to cluster the samples;
2. For each sample, produce a hierarchical clustering;
3. In a `(n_samples, n_samples)`-shaped matrix, store the number of times a sample pair appears together in a bootstrapped clustering.
4. Perform a hierarchical clustering on the resulting matrix.

This identifies groups of samples that frequently occur together in clusterings, regardless of the genes chosen.
Thus, these samples can be considered to robustly cluster together.

*Hint: use `np.random.choice` with `replacement=True` to create bootstrap samples of row indices.*

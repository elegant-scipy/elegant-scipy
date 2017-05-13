# Elegant NumPy: The Foundation of Scientific Python

> [NumPy] is everywhere. It is all around us. Even now, in this very room.
> You can see it when you look out your window or when you turn on your
> television. You can feel it when you go to work... when you go to church...
> when you pay your taxes.
>
> — Morpheus, *The Matrix*

This chapter touches on some statistical functions in SciPy, but more than that, it focuses on exploring the NumPy array, a data structure that underlies almost all numerical scientific computation in Python.
We will see how NumPy array operations enable concise and efficient code when manipulating numerical data.

Our use case is using gene expression data from The Cancer Genome Atlas (TCGA) project to predict mortality in skin cancer patients.
We will be working towards this goal throughout Chapters 1 and 2, learning about some key SciPy concepts along the way.
Before we can predict mortality, we will need to normalize the expression data using a method called RPKM normalization.
This allows the comparison of measurements between different samples and genes.
(We will unpack what "gene expression" means in just a moment.)

Let's start with a code snippet to tantalize, and motivate the ideas in this chapter.
As we will do in each chapter, we open with a code sample that we believe epitomizes the elegance and power of a particular function from the SciPy ecosystem.
In this case, we want to highlight NumPy's vectorization and broadcasting rules, which allow us to manipulate and reason about data arrays very efficiently.


```python
def rpkm(counts, lengths):
    """Calculate reads per kilobase transcript per million reads.

    RPKM = (10^9 * C) / (N * L)

    Where:
    C = Number of reads mapped to a gene
    N = Total mapped reads in the experiment
    L = Exon length in base pairs for a gene

    Parameters
    ----------
    counts: array, shape (N_genes, N_samples)
        RNAseq (or similar) count data where columns are individual samples
        and rows are genes.
    lengths: array, shape (N_genes,)
        Gene lengths in base pairs in the same order
        as the rows in counts.

    Returns
    -------
    normed : array, shape (N_genes, N_samples)
        The RPKM normalized counts matrix.
    """
    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts

    normed = 1e9 * C / (N[np.newaxis, :] * L[:, np.newaxis])

    return(normed)
```

This example illustrates some of the ways that NumPy arrays can make your code more elegant:

- Arrays can be one-dimensional, like lists, but they can also be two-dimensional, like matrices, and higher-dimensional still. This allows them to represent many different kinds of numerical data. In our case, we are manipulating a 2D matrix.
- Arrays can be operated on along *axes*. In the first line, we calculate the
  sum down each column by specifying `axis=0`.
- Arrays allow the expression of many numerical operations at once.
For example towards the end of the function we divide the 2D array of counts (C) by the 1D array of column sums (N).
This is broadcasting. More on how this works in just a moment!

Before we delve into the power of NumPy, let's spend some time to understand the biological data that we will be working with.

## What is gene expression?

We will work our way through a *gene expression analysis* to demonstrate the power of NumPy and SciPy to solve a real-world biological problem.
We will use the Pandas library, which builds on NumPy, to read and munge our data files, and then we manipulate our data efficiently in NumPy arrays.

The so-called [central dogma of molecular biology](https://en.wikipedia.org/wiki/Central_dogma_of_molecular_biology) states that all the information needed to run a cell (or an organism, for that matter) is stored in a molecule called *deoxyribonucleic acid*, or DNA.
This molecule has a repetitive backbone on which lie chemical groups called *bases*, in sequence.
There are four kinds of bases, abbreviated to A, C, G, and T, constituting an alphabet with which information is stored.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg"/>
<!-- caption text="The chemical structure of DNA. Image by Madeleine Price Ball, used under the terms of the CC0 public domain license" -->

To access this information, the DNA is *transcribed* into a sister molecule called *messenger ribonucleic acid*, or mRNA.
Finally, this mRNA is *translated* into proteins, the workhorses of the cell.
A section of DNA that encodes the information to make a protein (via mRNA) is called a gene.

The amount of mRNA produced from a given gene is called the *expression* of that gene.
Although we would ideally like to measure protein levels, this is a much harder task than measuring mRNA.
Fortunately, expression levels of an mRNA and levels of its corresponding protein are usually correlated ([Maier, Güell, and Serrano, 2009](http://www.sciencedirect.com/science/article/pii/S0014579309008126)).
Therefore, we usually measure mRNA levels and base our analyses on that.
As you will see below, it often doesn't matter, because we are using mRNA levels for their power to predict biological outcomes, rather than to make specific statements about proteins.

<img src="../figures/central_dogma.png"/>
<!-- caption text="Central Dogma of Molecular Biology" -->

It's important to note that the DNA in every cell of your body is identical.
Thus, the differences between cells arise from *differential expression* of
that DNA into RNA: in different cells, different parts of the DNA are processed
into downstream molecules. Similarly, as we shall see in this chapter and the
next, differential expression can distinguish different kinds of cancer.

<img src="../figures/differential_gene_expression.png"/>
<!-- caption text="Gene expression" -->

The state-of-the-art technology to measure mRNA is RNA sequencing (RNAseq).
RNA is extracted from a tissue sample, for example from a biopsy from a patient, *reverse transcribed* back into DNA (which is more stable), and then read out using chemically modified bases that glow when they are incorporated into the DNA sequence.
Currently, high-throughput sequencing machines can only read short fragments (approximately 100 bases is common). These short sequences are called “reads”.
We measure millions of reads and then based on their sequence we count how many reads came from each gene.
We’ll be starting directly from this count data.

<img src="../figures/RNAseq.png"/>
<!-- caption text="RNA sequencing (RNAseq)" -->

Here's an example of what this gene expression data looks like.

|        | Cell type A | Cell type B |
|--------|-------------|-------------|
| Gene 0 | 100         | 200         |
| Gene 1 | 50          | 0           |
| Gene 2 | 350         | 100         |

The data is a table of counts, integers representing how many reads were observed for each gene in each cell type.
See how the counts for each gene differ between the cell types?
We can use this information to tell us about the differences between these two types of cell.

One way to represent this data in Python would be as a list of lists:

```python
gene0 = [100, 200]
gene1 = [50, 0]
gene2 = [350, 100]
expression_data = [gene0, gene1, gene2]
```

Above, each gene's expression across different cell types is stored in a list of Python integers.
Then, we store all of these lists in a list (a meta-list, if you will).
We can retrieve individual data points using two levels of list indexing:

```python
expression_data[2][0]
```

It turns out that, because of the way the Python interpreter works, this is a very inefficient way to store these data points.
First, Python lists are always lists of *objects*, so that the above list `gene2` is not a list of integers, but a list of *pointers* to integers, which is unnecessary overhead.
Additionally, this means that each of these lists and each of these integers end up in a completely different, random part of your computer's RAM.
However, modern processors actually like to retrieve things from memory in *chunks*, so this spreading of the data throughout the RAM is inefficient.

This is precisely the problem solved by the *NumPy array*.

## NumPy N-dimensional arrays

One of the key NumPy data types is the N-dimensional array (ndarray, or just array).
Ndarrays underpin lots of awesome data manipulation techniques in SciPy.
In particular, we're going to explore vectorization and broadcasting,
techniques that allow us to write powerful, elegant code to manipulate our data.

First, let's get our heads around the the ndarray.
These arrays must be homogeneous: all items in an array must be the same type.
In our case we will need to store integers.
Ndarrays are called N-dimensional because they can have any number of dimensions.
A 1-dimensional array is roughly equivalent to a Python list:

```python
import numpy as np

array1d = np.array([1, 2, 3, 4])
print(array1d)
print(type(array1d))
```

Arrays have particular attributes and methods, that you can access by placing a dot after the array name.
For example, you can get the array's *shape*:

```python
print(array1d.shape)
```

Here, it's just a tuple with a single number.
You might wonder why you wouldn't just use `len`, as you would for a list.
That will work, but it doesn't extend to *two-dimensional* arrays.

This is what we use to represent our mini gene expression table from above:

```python
array2d = np.array(expression_data)
print(array2d)
print(array2d.shape)
print(type(array2d))
```

Now you can see that the `shape` attribute generalises `len` to account for the size of multiple dimensions of an array of data.

<img src="../figures/NumPy_ndarrays_v2.png"/>
<!-- caption text="Visualizing NumPy's ndarrays in one, two and three dimensions" -->

Arrays have other attributes, such as `ndim`, the number of dimensions:

```python
print(array2d.ndim)
```

You'll become familiar with all of these as you start to use NumPy more for your own data analysis.

NumPy arrays can represent data that has even more dimensions, such as magnetic resonance imaging (MRI) data, which includes measurements within a 3D volume.
If we store MRI values over time, we might need a 4D NumPy array.

For now, we'll stick to 2D data.
Later chapters will introduce higher-dimensional data and will teach you to write code that works for data of any number of dimensions.

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

# Create an ndarray of integers in the range
# 0 up to (but not including) 1,000,000
array = np.arange(1e6)

# Convert it to a list
list_array = array.tolist()
```

Let's compare how long it takes to multiply all the values in the array by 5,
using the IPython `timeit` magic function. First, when the data is in a list:

```python
%timeit -n10 y = [val * 5 for val in list_array]
```

Now, using NumPy's built-in *vectorized* operations:

```python
%timeit -n10 x = array * 5
```

Over 50 times faster, and more concise, too!

Arrays are also size efficient.
In Python, each element in a list is an object and is given a healthy memory allocation (or is that unhealthy?).
In contrast, in arrays, each element takes up just the necessary amount of memory.
For example, an array of 64-bit integers takes up exactly 64-bits per element, plus some very small overhead for array metadata, such as the `shape` attribute we discussed above.
This is generally much less than would be given to objects in a python list.
(If you're interested in digging into how Python memory allocation works, check out Jake VanderPlas' blog post, [Why Python is Slow: Looking Under the Hood](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/).)

Plus, when computing with arrays, you can also use *slices* that subset the array *without copying the underlying data*.

```python
# Create an ndarray x
x = np.array([1, 2, 3], np.int32)
print(x)
```

```python
# Create a "slice" of x
y = x[:2]
print(y)
```

```python
# Set the first element of y to be 6
y[0] = 6
print(y)
```

Notice that although we edited `y`, `x` has also changed, because `y` was referencing the same data!

```python
# Now the first element in x has changed to 6!
print(x)
```

This does mean you have to be careful with array references.
If you want to manipulate the data without touching the original, it's easy to make a copy:

```python
y = np.copy(x[:2])
```

### Vectorization

Earlier we talked about the speed of operations on arrays.
Once of the tricks Numpy uses to speed things up is *vectorization*.
Vectorization is where you apply a calculation to each element in an array, without having to use a for loop.
In addition to speeding things up, this can result in more natural, readable code.
Let's look at some examples.

```python
x = np.array([1, 2, 3, 4])
print(x * 2)
```

Here, we have `x`, an array of 4 values, and we have implicitly multiplied every element in `x` by 2, a single value.

```python
y = np.array([0, 1, 2, 1])
print(x + y)
```

Now, we have added together each element in `x` to its corresponding element in `y`, an array of the same shape.

Both of these operations are simple and, we hope, intuitive examples of vectorization.
NumPy also makes them very fast, much faster than iterating over the arrays manually.
(Feel free to play with this yourself using the `%timeit` IPython magic we saw earlier.)

### Broadcasting

One of the most powerful and often misunderstood features of ndarrays is broadcasting.
Broadcasting is a way of performing implicit operations between two arrays.
It allows you to perform operations on arrays of *compatible* shapes, to create arrays bigger than either of the starting ones.
For example, we can compute the [outer product](https://en.wikipedia.org/wiki/Outer_product) of two vectors, by reshaping them appropriately:

```python
x = np.array([1, 2, 3, 4])
x = np.reshape(x, (len(x), 1))
print(x)
```

```python
y = np.array([0, 1, 2, 1])
y = np.reshape(y, (1, len(y)))
print(y)
```

Two shapes are compatible when, for each dimension, either is equal to
1 (one) or they match one another[^more_dimensions].

[^more_dimensions]: We always start by comparing the last dimensions,
                    and work our way forward, ignoring excess
                    dimensions in the case of one array having more
                    than the other.  E.g., `(3, 5, 1)` and `(5, 8)`
                    would match.

Let's check the shapes of these two arrays.

```python
print(x.shape)
print(y.shape)
```

Both arrays have two dimensions and the inner dimensions of both arrays are 1, so the dimensions are compatible!

```python
outer = x * y
print(outer)
```

The outer dimensions tell you how size of the resulting array.
In our case we expect a (4, 4) array:

```python
print(outer.shape)
```

You can see for yourself that `outer[i, j] = x[i] * y[j]` for all `(i, j)`.

This was accomplished by NumPy's [broadcasting rules](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), which implicitly expand dimensions of size 1 in one array to match the corresponding dimension of the other array.
Don't worry, we will talk about these rules in more detail later in this chapter.

As we will see in the rest of the chapter, as we explore real data, broadcasting is extremely valuable to perform real-world calculations on arrays of data.
It allows us to express complex operations concisely and efficiently.

## Exploring a gene expression data set

The data set that we'll be using is an RNAseq experiment of skin cancer samples from The Cancer Genome Atlas (TCGA) project (http://cancergenome.nih.gov/).
We've already cleaned and sorted the data for you, so you can just use `data/counts.txt`
in the book repository.
In Chapter 2 we will be using this gene expression data to predict mortality in skin cancer patients, reproducing a simplified version of [Figures 5A and 5B](http://www.cell.com/action/showImagesData?pii=S0092-8674%2815%2900634-0) of a [paper](http://dx.doi.org/10.1016/j.cell.2015.05.044) from the TCGA consortium.
But first we need to get our heads around the biases in our data, and think about how we could improve it.

### Reading in the data with Pandas

We're first going to use Pandas to read in the table of counts.
Pandas is a Python library for data manipulation and analysis,
with particular emphasis on tabular and time series data.
Here, we will use it here to read in tabular data of mixed type.
It uses the DataFrame type, which is a flexible tabular format based on the data frame object in R.
For example the data we will read has a column of gene names (strings) and multiple columns of counts (integers), so reading it into a homogeneous array of numbers would be the wrong approach.
Although NumPy has some support for mixed data types (called "structured arrays"), it is not primarily designed for
this use case, which makes subsequent operations harder than they need to be.

By reading the data in as a Pandas DataFrame we can let Pandas do all the parsing, then extract out the relevant information and store it in a more efficient data type.
Here we are just using Pandas briefly to import data.
In later chapters we will see a bit more of Pandas, but for details, read *Python
for Data Analysis*, by Wes McKinney, creator of Pandas.

```python
import numpy as np
import pandas as pd

# Import TCGA melanoma data
filename = 'data/counts.txt'
with open(filename, 'rt') as f:
    data_table = pd.read_csv(f, index_col=0) # Parse file with pandas

print(data_table.iloc[:5, :5])
```

We can see that Pandas has kindly pulled out the header row and used it to name the columns.
The first column gives the name of each gene, and the remaining columns represent individual samples.

We will also need some corresponding metadata, including the sample information and the gene lengths.

```python
# Sample names
samples = list(data_table.columns)
```

We will need some information about the lengths of the genes for our normalization.
So that we can take advantage of some fancy pandas indexing, we're going to set
the index of the pandas table to be the gene names in the first column.

```python
# Import gene lengths
filename = 'data/genes.csv'
with open(filename, 'rt') as f:
    # Parse file with pandas, index by GeneSymbol
    gene_info = pd.read_csv(f, index_col=0)
print(gene_info.iloc[:5, :])
```

Let's check how well our gene length data matches up with our count data.

```python
print("Genes in data_table: ", data_table.shape[0])
print("Genes in gene_info: ", gene_info.shape[0])
```

There are more genes in our gene length data than were actually measured in the experiment.
Let's filter so we only get the relevant genes, and we want to make sure they are
in the same order as in our count data.
This is where pandas indexing comes in handy!
We can get the intersection of the gene names from our our two sources of data
and use these to index both data sets, ensuring they have the same genes in the same order.

```python
# Subset gene info to match the count data
matched_index = pd.Index.intersection(data_table.index, gene_info.index)
```

Now let's use the intersection of the gene names to index our count data.

```python
# 2D ndarray containing expression counts for each gene in each individual
counts = np.asarray(data_table.loc[matched_index], dtype=int)

gene_names = np.array(matched_index)

# Check how many genes and individuals were measured
print(f'{counts.shape[0]} genes measured in {counts.shape[1]} individuals.')
```

And our gene lengths.

```python
# 1D ndarray containing the lengths of each gene
gene_lengths = np.asarray(gene_info.loc[matched_index]['GeneLength'],
                          dtype=int)
```

And let's check the dimensions of our objects.

```python
print(counts.shape)
print(gene_lengths.shape)
```

As expected, they now match up nicely!

## Normalization

Before we do any kind of analysis with our data, it is important to take a look at it and determine if we need to normalize it first.
By normalize, we mean that we want to bring all our data onto the same scale so we can make a fair comparison.
We will consider two types of normalization commonly applied to expression data: between samples and between genes.
For example, when we consider differences between groups of patients, we want to know that they vary due to some biological difference, not just something technical.

### Between samples

For example, the number of counts for each individual can vary substantially in RNAseq experiments.
Let's take a look at the distribution of expression counts over all the genes.
First we will sum the rows to get the total counts of expression of all genes for each individual, so we can just look at the variation between individuals.
To visualize the distribution of total counts, we will use a kernel density estimation (KDE) function.
KDE is commonly used to smooth out histograms, which gives a clearer picture of the underlying distribution.

```python
# Make all plots appear inline in the Jupyter notebook from now onwards
%matplotlib inline
# Use our own style file for the plots
import matplotlib.pyplot as plt
plt.style.use('style/elegant.mplstyle')
# ignore MPL layout warnings
import warnings
warnings.filterwarnings('ignore', '.*Axes.*compatible.*tight_layout.*')
```

> **A quick note on plotting {.callout}**
>
> The code above does a few neat things to make our plots prettier.

> First, `%matplotlib inline` is a Jupyter notebook [magic
> command](http://ipython.org/ipython-doc/dev/interactive/tutorial.html#magics-explained),
> that simply makes all plots appear in the notebook rather than pop up a new
> window. If you are running a Jupyter notebook interactively, you can use
> `%matplotlib notebook` instead to get an interactive figure, rather than a
> static image of each plot.
>
> Second, we import `matplotlib.pyplot` then direct it to use our own plotting
> style `plt.style.use('style/elegant.mplstyle')`. You will see a block of code
> like this before the first plot in every chapter.
>
> You may have seen people importing existing styles like this:
> `plt.style.use('ggplot')`. But we wanted some particular settings, and we
> wanted all the plots in this book to follow the same style. So we rolled our
> own matplotlib style. To see how we did it, take a look at the style file in
> the Elegant SciPy repository: `style/elegant.mplstyle`. For more information
> on styles, check out the [Matplotlib documentation on style
> sheets](http://matplotlib.org/users/style_sheets.html).

Now back to plotting our counts distribution!

```python
total_counts = np.sum(counts, axis=0)  # sum columns together
                                       # (axis=1 would sum rows)

from scipy import stats

# Use Gaussian smoothing to estimate the density
density = stats.kde.gaussian_kde(total_counts)

# Make values for which to estimate the density, for plotting
x = np.arange(min(total_counts), max(total_counts), 10000)

# Make the density plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, density(x))
ax.set_xlabel("Total counts per individual")
ax.set_ylabel("Density")

plt.show()

print(f'Count statistics:\n  min:  {np.min(total_counts)}'
       '\n  mean: {np.mean(total_counts)}'
       '\n  max:  {np.max(total_counts)}')
```
<!-- caption text="Density plot of gene expression counts per individual using KDE smoothing" -->

We can see that there is an order of magnitude difference in the total number of counts between the lowest and the highest individual.
This means that a different number of RNAseq reads were generated for each individual.
We say that these individuals have different library sizes.

#### Normalizing library size between samples

Let's take a closer look at ranges of gene expression for each individual, so when
we apply our normalization we can see it in action. We'll subset a random sample
of just 70 columns to keep the plotting from getting too messy.

```python
# Subset data for plotting
np.random.seed(seed=7) # Set seed so we will get consistent results
# Randomly select 70 samples
samples_index = np.random.choice(range(counts.shape[1]), size=70, replace=False)
counts_subset = counts[:, samples_index]
```

```python
# Some custom x-axis labelling to make our plots easier to read
def reduce_xaxis_labels(ax, factor):
    """Show only every ith label to prevent crowding on x-axis
        e.g. factor = 2 would plot every second x-axis label,
        starting at the first.

    Parameters
    ----------
    ax : matplotlib plot axis to be adjusted
    factor : int, factor to reduce the number of x-axis labels by
    """
    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    for label in ax.xaxis.get_ticklabels()[::factor]:
        label.set_visible(True)
```

```python
# Bar plot of expression counts by individual
fig, ax = plt.subplots(figsize=(16,5))

ax.boxplot(counts_subset)
ax.set_title("Gene expression counts raw")
ax.set_xlabel("Individuals")
ax.set_ylabel("Gene expression counts")
reduce_xaxis_labels(ax, 2)
```
<!-- caption text="Boxplot of gene expression counts per individual" -->

There are obviously a lot of outliers at the high expression end of the scale and a lot of variation between individuals, but pretty hard to see because everything is clustered around zero.
So let's do log(n + 1) of our data so it's a bit easier to look at.
Both the log function and the n + 1 step can be done using broadcasting to simplify our code and speed things up.

```python
# Bar plot of expression counts by individual
fig, ax = plt.subplots(figsize=(16,5))

ax.boxplot(np.log(counts_subset + 1))
ax.set_title("Gene expression counts raw")
ax.set_xlabel("Individuals")
ax.set_ylabel("log gene expression counts")
reduce_xaxis_labels(ax, 2)
```
<!-- caption text="Boxplot of gene expression counts per individual (log scale)" -->

Now let's see what happens when we normalize by library size.

```python
# Normalize by library size
# Divide the expression counts by the total counts for that individual
# Multiply by 1 million to get things back in a similar scale
counts_lib_norm = counts / total_counts * 1000000
# Notice how we just used broadcasting twice there!
counts_subset_lib_norm = counts_lib_norm[:,samples_index]

# Bar plot of expression counts by individual
fig, ax = plt.subplots(figsize=(16,5))

ax.boxplot(np.log(counts_subset_lib_norm + 1))
ax.set_title("Gene expression counts normalized by library size")
ax.set_xlabel("Individuals")
ax.set_ylabel("log gene expression counts")
reduce_xaxis_labels(ax, 2)
```
<!-- caption text="Boxplot of library-normalized gene expression counts per individual (log scale)" -->

Much better!
Also notice how we used broadcasting twice there.
Once to divide all the gene expression counts by the total for that column, and then again to multiply all the values by 1 million.

Finally, let's compare our normalized data to the raw data.

```python
import itertools as it
from collections import defaultdict


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
        default palette contains six colors.)

    Other parameters
    ----------------
    kwargs : dict
        Keyword arguments to pass on to `plt.boxplot`.
    """
    all_classes = sorted(set(classes))
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class2color = dict(zip(all_classes, it.cycle(colors)))

    # map classes to data vectors
    # other classes get an empty list at that position for offset
    class2data = defaultdict(list)
    for distrib, cls in zip(data, classes):
        for c in all_classes:
            class2data[c].append([])
        class2data[cls][-1] = distrib

    # then, do each boxplot in turn with the appropriate color
    fig, ax = plt.subplots(figsize=(2 * len(data), 5))
    lines = []
    for cls in all_classes:
        # set color for all elements of the boxplot
        for key in ['boxprops', 'whiskerprops', 'capprops',
                    'medianprops', 'flierprops']:
            kwargs.setdefault(key, {}).update(color=class2color[cls])
        # draw the boxplot
        box = ax.boxplot(class2data[cls], **kwargs)
        lines.append(box['whiskers'][0])
    ax.legend(lines, all_classes)
    return ax
```

Now we can plot a colored boxplot according to normalized vs unnormalized samples.
We show only three samples from each class for illustration:

```python
log_counts_3 = list(np.log(counts.T[:3] + 1))
log_ncounts_3 = list(np.log(counts_lib_norm.T[:3] + 1))
ax = class_boxplot(log_counts_3 + log_ncounts_3,
                   ['raw counts'] * 3 + ['normalized by library size'] * 3,
                   labels=[1, 2, 3, 1, 2, 3])
ax.set_xlabel('sample number')
ax.set_ylabel('log gene expression counts');
```
<!-- caption text="Comparing raw and library normalized gene expression counts in three samples (log scale)" -->

You can see that the normalized distributions are a little bit more similar
once we have taken library size (the sum of those distributions) into account.
Now we are comparing like with like between the samples!
But what about differences between the genes?

### Between genes

We can also get into some strife when trying to compare different genes.
The number of counts for a gene, is related to the gene length.
Let's say we have gene A and gene B.
Gene B is twice as long as gene A.
Both are expressed at similar levels in the sample, i.e. both produce a similar number of mRNA molecules.
Therefore you would expect that gene B would have about twice as many counts as gene A.
Remember, that when we do an RNAseq experiment, we are fragmenting the transcript, and sampling reads from that pool of fragments.
The counts are the number of reads from that gene in a given sample.
So if a gene is twice as long, we are twice as likely to sample it.
If we want to compare between genes we will have to do some more normalization.

<img src="../figures/gene_length_counts.png"/>
<!-- caption text="Relationship between counts and gene length" -->

Let's see if the relationship between gene length and counts plays out in our data set.

```python
def binned_boxplot(x, y, *,  # check out this Python 3 exclusive! (*see tip box)
                   xlabel='gene length (log scale)',
                   ylabel='average log counts'):
    """Plot the distribution of `y` dependent on `x` using many boxplots.

    Note: all inputs are expected to be log-scaled.

    Parameters
    ----------
    x: 1D array of float
        Independent variable values.
    y: 1D array of float
        Dependent variable values.
    """
    # Define bins of `x` depending on density of observations
    x_hist, x_bins = np.histogram(x, bins='auto')

    # Use `np.digitize` to number the bins
    # Discard the last bin edge because it breaks the right-open assumption
    # of `digitize`. The max observation correctly goes into the last bin.
    x_bin_idxs = np.digitize(x, x_bins[:-1])

    # Use those indices to create a list of arrays, each containing the `y`
    # values corresponding to `x`s in that bin. This is the input format
    # expected by `plt.boxplot`
    binned_y = [y[x_bin_idxs == i]
                for i in range(np.max(x_bin_idxs))]
    fig, ax = plt.subplots(figsize=(16,3))

    # Make the x-axis labels using the bin centers
    x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
    x_ticklabels = np.round(np.exp(x_bin_centers)).astype(int)

    # make the boxplot
    ax.boxplot(binned_y, labels=x_ticklabels)

    # show only every 10th label to prevent crowding on x-axis
    reduce_xaxis_labels(ax, 10)

    # Adjust the axis names
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
```





> **Python 3 Tip: using `*` to create keyword-only arguments {.callout}**
>
> Since version 3.0 Python allows
> ["keyword-only" arguments](https://www.python.org/dev/peps/pep-3102/).
> These are arguments that you have to call using a keyword, rather than relying
> on position alone.
> For example, with the `binned_boxplot` function we just wrote, you can call it
> like this:
>
>     >>> binned_boxplot(x, y, xlabel='my x label', ylabel='my y label')
>
> but not this, which would have been valid Python 2, but raises an error in
> Python 3:
>
>     >>> binned_boxplot(x, y, 'my x label', 'my y label')
>
>     ---------------------------------------------------------------------------
>     TypeError                                 Traceback (most recent call last)
>     <ipython-input-58-7a118d2d5750in <module>()
>         1 x_vals = [1, 2, 3, 4, 5]
>         2 y_vals = [1, 2, 3, 4, 5]
>     ----3 binned_boxplot(x, y, 'my x label', 'my y label')
>
>     TypeError: binned_boxplot() takes 2 positional arguments but 4 were given
>
> The idea is to prevent you from accidentally doing something like this:
>
>     binned_boxplot(x, y, 'my y label')
>
> which would give you your y label on the x axis, and is a common error for
> signatures with many optional parameters that don't have an obvious ordering.

```python
log_counts = np.log(counts_lib_norm + 1)
mean_log_counts = np.mean(log_counts, axis=1)  # across samples
log_gene_lengths = np.log(gene_lengths)
```

```python
binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
```
<!-- caption text="The relationship between gene length and average expression (log scale)" -->

We can see that the longer a gene is, the higher its measured counts! As
explained above, this is an artifact of the technique, not a biological signal!
How do we account for this?

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

Now let's implement RPKM over the entire counts array.

```python
# Make our variable names the same as the RPKM formula so we can compare easily
C = counts
N = counts.sum(axis=0)  # sum each column to get total reads per sample
L = gene_lengths  # lengths for each gene, matching rows in `C`
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

#### Broadcasting rules

Broadcasting allows calculations between ndarrays that have differing shapes.
Numpy uses broadcasting rules to make these manipulations a little easier.
For example, if the input arrays do not have the same number of dimensions,
then then an additional dimension is added to the start of the first array,
with a value of 1.
Once the two arrays have the same number of dimensions,
broadcasting can only occur if the sizes of the dimensions match,
or one of them is equal to 1.

For example, let's say we have two ndarrays, A and B:
`A.shape = (1, 2)`
`B.shape = (2,)`

If we performed the operation `A * B` then broadcasting would occur.
B has fewer dimension than A, so during the calculation
a new dimension is prepended to B with value 1.
`B.shape = (1, 2)`
Now A and B have the same number of dimensions, so broadcasting can proceed.

Now let's say we have another array, C:

`C.shape = (2, 1)`
`B.shape = (2,)`

Now, if we were to do the operation `C * B`,
a new dimension needs to be prepended to B.

`B.shape = (1, 2)`

However, the dimensions of the two ndarrays do not match,
so broadcasting will fail.

Let's say that we know that it is appropriate to broadcast B over C.
We can explicitly add a new dimension to B using `np.newaxis`.
Let's see this in our normalization by RPKM.

Let's look at the dimensions of our arrays.

```python
print('C_tmp.shape', C_tmp.shape)
print('L.shape', L.shape)
```

We can see that `C_tmp` has 2 dimensions, while L has one.
So during broadcasting, an additional dimension will be prepended to L.
Then we will have:
```
C_tmp.shape (20500, 375)
L.shape (1, 20500)
```

The dimensions won't match!
We want to broadcast L over the first dimension of `C_tmp`,
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

Finally we need to normalize by the library size,
the total number of counts for that column.
Remember that we have already calculated N with:

```
N = counts.sum(axis=0) # sum each column to get total reads per sample
```

```python
# Check the shapes of C_tmp and N
print('C_tmp.shape', C_tmp.shape)
print('N.shape', N.shape)
```

Once we trigger broadcasting, an additional dimension will be
prepended to N:

`N.shape (1, 375)`

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
def rpkm(counts, lengths):
    """Calculate reads per kilobase transcript per million reads.

    RPKM = (10^9 * C) / (N * L)

    Where:
    C = Number of reads mapped to a gene
    N = Total mapped reads in the experiment
    L = Exon length in base pairs for a gene

    Parameters
    ----------
    counts: array, shape (N_genes, N_samples)
        RNAseq (or similar) count data where columns are individual samples
        and rows are genes.
    lengths: array, shape (N_genes,)
        Gene lengths in base pairs in the same order
        as the rows in counts.

    Returns
    -------
    normed : array, shape (N_genes, N_samples)
        The RPKM normalized counts matrix.
    """
    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts

    normed = 1e9 * C / (N[np.newaxis, :] * L[:, np.newaxis])

    return(normed)
```

```python
counts_rpkm = rpkm(counts, gene_lengths)
```

#### RPKM between gene normalization

Let's see the RPKM normalization's effect in action. First, as a reminder, here's
the distribution of mean log counts as a function of gene length:

```python
log_counts = np.log(counts + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)

binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
```
<!-- caption text="The relationship between gene length and average expression before RPKM normalization (log scale)" -->

Now, the same plot with the RPKM-normalized values:

```python
log_counts = np.log(counts_rpkm + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)

binned_boxplot(x=log_gene_lengths, y=mean_log_counts)
```

<!-- caption text="The relationship between gene length and average expression after RPKM normalization (log scale)" -->

You can see that the mean expression counts have flattened quite a bit,
especially for genes larger than about 3,000 base pairs.
(Smaller genes still appear to have low expression — these may be too small for
the statistical power of the RPKM method.)

RPKM normalization can be useful to compare the expression profile of different genes.
We've already seen that longer genes have higher counts, but this doesn't mean their expression level is actually higher.
Let's choose a short gene and a long gene and compare their counts before and after RPKM normalization to see what we mean.

```python
gene_idxs = np.array([80, 186])
gene1, gene2 = gene_names[gene_idxs]
len1, len2 = gene_lengths[gene_idxs]
gene_labels = [f'{gene1}, {len1}bp', f'{gene2}, {len2}bp']

log_counts_2 = list(np.log(counts[gene_idxs] + 1))
log_ncounts_2 = list(np.log(counts_rpkm[gene_idxs] + 1))

ax = class_boxplot(log_counts_2,
                   ['raw counts'] * 3,
                   labels=genes2_labels)
ax.set_xlabel('Genes')
ax.set_ylabel('log gene expression counts over all samples');
```
<!-- caption text="Comparing expression of two genes before RPKM normalization" -->

If we look just at the raw counts, it looks like the longer Gene B is expressed
slightly more than Gene A.
But, after RPKM normalization, a different picture emerges:

```python
ax = class_boxplot(log_ncounts_2,
                   ['RPKM normalized'] * 3,
                   labels=genes2_labels)
ax.set_xlabel('Genes')
ax.set_ylabel('log RPKM gene expression counts over all samples');
```
<!-- caption text="Comparing expression of two genes after RPKM normalization" -->

Now it looks like gene A is actually expressed at a much higher level than gene B.
This is because RPKM includes normalization for gene length, so we can now directly compare between genes of different lengths.

## Taking stock

So far we have:
- imported data using Pandas;
- gotten to know the key NumPy object class: the ndarray; and
- used the power of broadcasting to make our calculations more elegant.

In Chapter 2 we will continue working with the same data set, implementing a
more sophisticated normalization technique, then using clustering to make some
predictions about mortality in skin cancer patients.

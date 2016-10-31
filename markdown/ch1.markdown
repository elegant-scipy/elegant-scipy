# Elegant NumPy: The Foundation of Scientific Python

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

    counts: 2D numpy ndarray (numerical)
        RNAseq (or similar) count data where columns are individual samples
        and rows are genes.
    lengths: list or 1D numpy ndarray (numerical)
        Gene lengths in base pairs in the same order
        as the rows in counts.
    """

    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts

    rpkm = ( (10e9 * C) / N[np.newaxis, :] ) / L[:, np.newaxis]

    return(rpkm)
```

This example illustrates some of the ways that NumPy arrays can make your code more elegant:

- Arrays can be one-dimensional, like lists, but they can also be two-dimensional, like matrices, and higher-dimensional still. This allows them to represent many different kinds of numerical data. In our case, we are manipulating a 2D matrix.
- Arrays can be operated on along *axes*. In the first line, we calculate the sum down each column by specifying a *different* `axis`.
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

![The chemical structure of DNA](https://upload.wikimedia.org/wikipedia/commons/e/e4/DNA_chemical_structure.svg)
*Image by Madeleine Price Ball, used under the terms of the CC0 public domain license*

To access this information, the DNA is *transcribed* into a sister molecule called *messenger ribonucleic acid*, or mRNA.
Finally, this mRNA is *translated* into proteins, the workhorses of the cell.
A section of DNA that encodes the information to make a protein (via mRNA) is called a gene.

The amount of mRNA produced from a given gene is called the *expression* of that gene.
Although we would ideally like to measure protein levels, this is a much harder task than measuring mRNA.
Fortunately, expression levels of an mRNA and levels of its corresponding protein are usually correlated ([Maier, Güell, and Serrano, 2009](http://www.sciencedirect.com/science/article/pii/S0014579309008126)).
Therefore, we usually measure mRNA levels and base our analyses on that.
As you will see below, it often doesn't matter, because we are using mRNA levels for their power to predict biological outcomes, rather than to make specific statements about proteins.

![Central Dogma of Molecular Biology](../figures/central_dogma.png)
**[ED NOTE doodle by Juan Nunez-Iglesias]**

It's important to note that the DNA in every cell of your body is identical.
Thus, the differences between cells arise from *differential expression* of that DNA into RNA.
Similarly, as we shall see in this chapter, differential expression can distinguish different kinds of cancer.

![Gene expression](../figures/differential_gene_expression.png)
**[ED NOTE doodle by Juan Nunez-Iglesias]**

The state-of-the-art technology to measure mRNA is RNA sequencing (RNAseq).
RNA is extracted from a tissue sample, for example from a biopsy from a patient, *reverse transcribed* back into DNA (which is more stable), and then read out using chemically modified bases that glow when they are incorporated into the DNA sequence.
Currently, high-throughput sequencing machines can only read short fragments (approximately 100 bases is common). These short sequences are called “reads”.
We measure millions of reads and then based on their sequence we count how many reads came from each gene.
For this chapter we’ll be starting directly from this count data, but in [ch7?] we will talk more about how this type of data can be determined.

![RNAseq](../figures/RNAseq.png)
**[ED NOTE doodle by Juan Nunez-Iglesias]**

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
However, modern processors actually like to retrieve things from memory in *chunks*, so this spreading of the data throughout the RAM is very bad.

This is precisely the problem solved by the *NumPy array*.

## NumPy N-dimensional arrays

One of the key NumPy data types is the N-dimensional array (ndarray, or just array).
Arrays must be homogeneous; all items in an array must be the same type.
In our case we will need to store integers.

Ndarrays are called N-dimensional because they can have any number of dimensions.
A 1-dimesional array is roughly equivalent to a Python list:

```python
import numpy as np

one_d_array = np.array([1,2,3,4])
print(one_d_array)
print(type(one_d_array))
```

Arrays have particular attributes and methods, that you can access by placing a dot after the array name.
For example, you can get the array's *shape*:

```python
print(one_d_array.shape)
```

Here, it's just a tuple with a single number.
You might wonder why you wouldn't just use `len`, as you would for a list.
That will work, but it doesn't extend to *two-dimensional* arrays.

This is what we use to represent our mini gene expression table from above:

```python
two_d_array = np.array(expression_data)
print(two_d_array.shape)
```

Now you can see that the `shape` attribute generalises `len` to account for the size of multiple dimensions of an array of data.

![multi-dimensional array diagram](../figures/NumPy_ndarrays.png)
**[ED NOTE doodle by Juan Nunez-Iglesias]**

Arrays have other attributes, such as `ndim`, the number of dimensions:

```python
print(two_d_array.ndim)
```

You'll become familiar with all of these as you start to use NumPy more for your own data analysis.

NumPy arrays can represent data that has even more dimensions, such as magnetic resonance imaging (MRI) data, which includes measurements within a 3D volume.
If we store MRI values over time, we might need a 4D NumPy array.

For now, we'll stick to 2D data.
Later chapters will introduce higher-dimensional data and will teach you to write code that works for data of any number of dimensions!

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
# 0 up to (but not including) 10,000,000
nd_array = np.arange(1e6)
# Convert arr to a list
list_array = nd_array.tolist()
```

```python
%%timeit -n10
# Time how long it takes to multiply each element in the list by 5
for i, val in enumerate(list_array):
    list_array[i] = val * 5
```

```python
%%timeit -n10
# Use the IPython "magic" command timeit to time how
# long it takes to multiply each element in the ndarray by 5
x = nd_array * 5
```

Close to 100 times faster, and more concise, too!

Arrays are also size efficient.
In Python, each element in a list is an object and is given a healthy memory allocation (or is that unhealthy?).
In contrast, in arrays, each element takes up just the necessary amount of memory.
For example, an array of 64-bit integers takes up exactly 64-bits per element, plus some very small overhead for array metadata, such as the `shape` attribute we discussed above.
This is generally much less than would be given to objects in a python list.
(If you're interested in digging into how Python memory allocation works, check out Jake VanderPlas' blog post [Why Python is Slow: Looking Under the Hood](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/).)

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
(Feel free to play with this yourself using the `%%timeit` IPython magic.)


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

In order to do broadcasting, the two arrays have to have the same number of dimensions and the sizes of the dimensions need to match (or be equal to 1).
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
In Chapter 2 we will be using this gene expression data to predict mortality in skin cancer patients, reproducing a simplified version of [Figures 5A and 5B](http://www.cell.com/action/showImagesData?pii=S0092-8674%2815%2900634-0) of a [paper](http://dx.doi.org/10.1016/j.cell.2015.05.044) from the TCGA consortium.
But first we need to get our heads around the biases in our data, and think about how we could improve it.

### Downloading the data

[Links to data!]

We're first going to use Pandas to read in the table of counts.
Pandas is a Python library for data manipulation and analysis,
with particular emphasis on tabular and time series data.
Here, we will use it here to read in tabular data of mixed type.
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

We can see that Pandas has kindly pulled out the header row and used it to name the columns.
The first column gives the name of the gene, and the remaining columns represent individual samples.

We will also needs some corresponding metadata, including the sample information and the gene lengths.

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
    gene_info = pd.read_csv(f, index_col=0) # Parse file with pandas, index by GeneSymbol
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
#Subset gene info to match the count data
matched_index = pd.Index.intersection(data_table.index, gene_info.index)
```

Now let's use the intersection of the gene names to index our count data.

```python
# 2D ndarray containing expression counts for each gene in each individual
counts = np.asarray(data_table.loc[matched_index], dtype=int)

# Check how many genes and individuals were measured
print("{0} genes measured in {1} individuals".format(counts.shape[0], counts.shape[1]))
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

### Between samples

For example, the number of counts for each individual can vary substantially in RNAseq experiments.
Let's take a look at the distribution of expression counts over all the genes.
First we will sum the rows to get the total counts of expression of all genes for each individual, so we can just look at the variation between individuals.
To visualize the distribution of total counts, we will use a kernel density estimation (KDE) function.
KDE is commonly used to smooth out histograms, which gives a clearer picture of the underlying distribution.

```python
%matplotlib inline
# Make all plots appear inline in the Jupyter notebook from now onwards

import matplotlib.pyplot as plt
plt.style.use('ggplot') # Use ggplot style graphs for something a little prettier
```

```python
total_counts = counts.sum(axis=0) # sum each column (axis=1 would sum rows)

from scipy import stats
density = stats.kde.gaussian_kde(total_counts) # Use gaussian smoothing to estimate the density
x = np.arange(min(total_counts), max(total_counts), 10000) # create ndarray of integers from min to max in steps of 10,000
plt.plot(x, density(x))
plt.xlabel("Total counts per individual")
plt.ylabel("Density")
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

**[ED'S NOTE]: the following function will probably be replaced by Seaborn's new boxplot function, which supports exactly this use case.**

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
Remember, that when we do an RNAseq experiment, we are fragmenting the transcript, and sampling reads from that pool of fragments.
The counts are the number of reads from that gene in a given sample.
So if a gene is twice as long, we are twice as likely to sample it.

![Relationship between counts and gene length](../figures/gene_length_counts.png)
**[ED NOTE doodle by Juan Nunez-Iglesias]**

Let's see if the relationship between gene length and counts plays out in our data set.

```python
def binned_boxplot(x, y,
                   xlabel='gene length (log scale)',
                   ylabel='average log counts'):
    """Plot the distribution of `y` dependent on `x` using many boxplots.

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
    plt.figure(figsize=(16,3))

    # Make the x-axis labels using the bin centers
    x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
    x_labels = np.round(np.exp(x_bin_centers)).astype(int)

    # use only every 5th label to prevent crowding on x-axis ticks
    labels = []
    for i, lab in enumerate(x_labels):
        if i % 5 == 0:
            labels.append(str(lab))
        else:
            labels.append('')
    # make the boxplot
    plt.boxplot(binned_y, labels=labels, sym=".")

    # Adjust the axis names
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
Now A and B have the same number of dimensions, so broadcasting can proceed.

Now let's say we have another array, C:  
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
def rpkm(counts, lengths):
    """Calculate reads per kilobase transcript per million reads.
    RPKM = (10^9 * C) / (N * L)

    Where:
    C = Number of reads mapped to a gene
    N = Total mapped reads in the experiment
    L = Exon length in base pairs for a gene

    counts: 2D numpy ndarray (numerical)
        RNAseq (or similar) count data where columns are individual samples
        and rows are genes.
    lengths: list or 1D numpy ndarray (numerical)
        Gene lengths in base pairs in the same order
        as the rows in counts.
    """

    N = np.sum(counts, axis=0) # sum each column to get total reads per sample
    L = lengths
    C = counts

    rpkm = ( (10e9 * C) / N[np.newaxis, :] ) / L[:, np.newaxis]

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

genes2_idx = [80, 186]
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

Just looking at the raw counts, it looks like the longer gene B is expressed slightly more than gene A.
Yet once we normalize to RPKM values, the story changes substantially.
Now it looks like gene A is actually expressed at a much higher level than gene B.
This is because RPKM includes normalization for gene length, so we can now directly compare between genes of dramatically different lengths.

## Taking stock

So far we have, imported data using Pandas, gotten to know the key NumPy data type: the ndarray, and used the power of broadcasting to make our calculations more elegant.

In Chapter 2 we will continue working with the same data set, implementing a more sophisticated normalization technique, and then using clustering to make some predictions about mortality in skin cancer patients.


# 2D Array Manipulation for RNAseq

## Gene Expression

In this chapter we’re going to work our way through a gene expression analysis to demonstrate the power of SciPy to solve a real-world biological problem that I come across everyday.
Along the way we will use Pandas to parse tabular data, and then manipulate our data efficiently in Numpy ndarrays.

But before we get to the juicy code, let me fill you in about my particular biological problem.
The central dogma of genetics says that all the information to run a cell is stored in the DNA.
To access this information, the DNA needs to be transcribed into messenger RNA (mRNA).
The amount of mRNA produced from a given gene is called the “expression” of that gene.
The mRNA is in turn is translated into protein, which is the workhorse of the cell.
Proteins can act as building blocks like the keratin that gives structure to your nails and hair, or the enzymes that allow you to digest your food.
Unfortunately, protein is particularly difficult to detect experimentally, but mRNA is actually pretty easy to measure.
So we make the assumption that if we measure the amount of mRNA we can gain insight into how the cells are functioning.

![Central Dogma of Biology](http://www.phschool.com/science/biology_place/biocoach/images/transcription/centdog.gif)
Note, this is an example image only, we have not checked license.

![Gene expression](http://www.ncbi.nlm.nih.gov/Class/MLACourse/Original8Hour/Genetics/cgap_conceptual_tour1.gif)
Note, this is an example image only, we have not checked license.

Currently, the most sensitive way to measure mRNA is to do an RNA sequencing (RNAseq) experiment. To do this we isolate all the mRNA from a sample, then we sequence it.
Currently, high-throughput sequencing machines can only read short fragments (approximately 100 bases is common). These short sequences are called “reads”.
We measure millions of reads and then based on their sequence we count how many reads came from each gene.
For this chapter we’ll be starting directly from this count data, but in [ch7?] we will talk more about how this type of data can be determined.

![RNAseq](http://bio.lundberg.gu.se/courses/vt13/rna4.JPG.jpg)
Note, this is an example image only, we have not checked license.

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
import numpy as np
import pandas as pd

# Import TCGA melanoma data
filename = 'data/counts.txt'
with open(filename, 'rt') as f:
    data_table = pd.read_csv(f, index_col=0) # Parse file with pandas

print(data_table.iloc[:5, :5])
```

We can see that Pandas has kindly pull out the header row and used it to name the columns.
The first three columns are information about a gene.
The ID of the gene, what chromosome it is on, and how long the gene is.
The remaining columns are IDs for the individual people who were tested, along with the name of the lab that performed the testing (Argonne National Laboratory for the first few).
Let's extract out the data that we need in a more useful format.


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
Both the log function and the n + 1 step can be done using broadcasting to simpify our code and speed things up.

```python
# Bar plot of expression counts by individual
plt.figure(figsize=(16,5))
plt.boxplot(np.log(counts_subset + 1), sym=".")
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
counts_subset_lib_norm = counts_lib_norm[:,samples_index]

# Bar plot of expression counts by individual
plt.figure(figsize=(16,5))
plt.boxplot(np.log(counts_subset_lib_norm + 1), sym=".")
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
log_counts_3 = list(np.log(counts.T[:3] + 1))
log_ncounts_3 = list(np.log(counts_lib_norm.T[:3] + 1))
class_boxplot(log_counts_3 + log_ncounts_3,
              ['raw counts'] * 3 + ['normalized by library size'] * 3,
              labels=[1, 2, 3, 1, 2, 3])
plt.xlabel('sample number')
plt.ylabel('log gene expression counts')
plt.show()
```

An example of the types of plots I'd like to show:
http://www.nature.com/nbt/journal/v32/n9/images_article/nbt.2931-F2.jpg

#### Between genes

Number of reads related to length of gene

```python
mean_counts = np.mean(counts_lib_norm, axis=1)  # mean expression per gene
plt.figure()
plt.scatter(gene_lengths, mean_counts)
plt.xlabel("Gene length in base pairs")
plt.ylabel("Mean expression counts for that gene")
#plt.xlim(0,40000)
#plt.ylim(0,10000)
plt.show()
```

Boxplot binned by gene length:

```python
log_counts = np.log(counts_lib_norm + 1)
mean_log_counts = np.mean(log_counts, axis=1)
log_gene_lengths = np.log(gene_lengths)
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


# Before normalisation
log_counts = np.log(counts + 1)
plot_col_density(log_counts, xlabel="Log count distribution for each individual")
```

Given an expression matrix (microarray data, read counts, etc) of ngenes by nsamples, quantile normalization ensures all samples have the same spread of data (by construction). It involves:

(optionally) log-transforming the data
sorting all the data points column-wise
averaging the rows
replacing each column quantile with the quantile of the average column.
This can be done with numpy and scipy easily and efficiently.
Assume we've read in the input matrix as X:

```python
import numpy as np
from scipy import stats

def quantile_norm(X):
    """ Given an expression matrix (microarray data, read counts, etc) of ngenes
    by nsamples, quantile normalization ensures all samples have the same spread
    of data (by construction).
    The data is first log transformed. The rows are averaged and each column
    quantile is replaced with the quantile of the average column.

    Parameters
    ----------
    X : 2D ndarray of counts
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
# After normalisation
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

    #construct your numpy array of data
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

## Heatmap

```python
def median_absolute_deviation(a, axis=None):
    """Compute Median Absolute Deviation of an ndarray along given axis.
    From http://informatique-python.readthedocs.org/en/latest/Exercices/mad.html
    """
    med = np.median(a, axis=axis)
    med = np.expand_dims(med, axis or 0)
    mad = np.median(np.abs(a - med), axis=axis)
    return mad


def most_variable_rows(data, n=1500, axis=1, method='mad'):
    """Subset data to the n most variable rows

    In this case, we want the n most variable genes.

    Parameters
    ----------
    data : 2D array of float
        The data to be subset
    n : int, optional
        Number of rows to return.
    axis : {0, 1}, optional
        The axis along which to compute variability
    method : {'mad', 'var'}, optional
        Use MAD (median absolute deviation) or variance to compute
        variability.
    """
    # compute variance along the axis with the chosen method
    # e.g. variance of each gene over the samples
    if method == 'mad':
        rowvar = median_absolute_deviation(data, axis=axis)
    elif method == 'var':
        rowvar = np.var(data, axis=axis)

    # Get sorted indices (ascending order), take the last n
    sort_indices = np.argsort(rowvar)[-n:]

    # use as index for data
    variable_data = data[sort_indices,:]

    return variable_data
```

```python
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform

def bicluster(data, linkage_method='average',
              n_clusters_r=10, n_clusters_c=3, distance_metric='correlation'):
    """Perform a biclustering, plot a heatmap with dendograms on each axis.

    Parameters
    ----------
    data : 2D ndarray
        The input data to bicluster.
    linkage_method : string, optional
        Method to be passed to `linkage`.
    n_clusters_r, n_clusters_c : int, optional
        Number of clusters for rows and columns.
    distance_metric : string, optional
        Distance metric to use for clustering. Anything accepted by
        `pdist` is acceptable here.

    Returns
    -------
    y_rows : linkage matrix
        The clustering of the rows of the input data.
    y_cols : linkage matrix
        The clustering of the cols of the input data.
    """
    fig = plt.figure(figsize=(8, 8))

    # Compute and plot row-wise dendogram
    # `add_axes` takes a "rectangle" input to add a subplot to a figure.
    # The figure is considered to have side-length 1 on each side, and its
    # bottom-left corner is at (0, 0).
    # The measurements passed to `add_axes` are the left, bottom, width, and
    # height of the subplot. Thus, to draw the left dendogram (for the rows),
    # we create a rectangle whose bottom-left corner is at (0.09, 0.1), and
    # measuring 0.2 in width and 0.6 in height.
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    y_rows = linkage(data, method=linkage_method, metric=distance_metric)
    # For a given number of clusters, we can obtain a cut of the linkage
    # tree by looking at the corresponding distance annotation in the linkage
    # matrix.
    threshold_r = (y_rows[-n_clusters_r, 2] + y_rows[-n_clusters_r+1, 2]) / 2
    z_rows = dendrogram(y_rows, orientation='right',
                        color_threshold=threshold_r)

    # Compute and plot column-wise dendogram
    # See notes above for explanation of parameters to `add_axes`
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    y_cols = linkage(data.T, method=linkage_method, metric=distance_metric)
    threshold_c = (y_cols[-n_clusters_c, 2] + y_cols[-n_clusters_c+1, 2]) / 2
    z_cols = dendrogram(y_cols, color_threshold=threshold_c)

    # Hide axes labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot data heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])

    # Sort data by the dendogram leaves
    idx_rows = leaves_list(y_rows)
    data = data[idx_rows, :]
    idx_cols = leaves_list(y_cols)
    data = data[:, idx_cols]

    im = axmatrix.matshow(data, aspect='auto', origin='lower', cmap='YlGnBu_r')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Axis labels
    plt.xlabel('Samples')
    plt.ylabel('Genes', labelpad=125)

    # Plot legend
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    plt.show()

    return y_rows, y_cols
```

```python
def most_variable_heatmap(counts):
    counts_log = np.log(counts + 1)
    counts_variable = most_variable_rows(counts_log, n=1500, method='var')
    yr, yc = bicluster(counts_variable)
    return yr, yc

yr, yc = most_variable_heatmap(quantile_norm(counts))
```

We can see that the data naturally fall into NUMCLUSTERS clusters.
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

To obtain a survival curve from survival times, we merely create a step function that decreases by $1/n$ at each step, where $n$ is the population size.
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
We leave you the exercise of implementing the approach described in the paper:

1. Take bootstrap samples (random choice with replacement) of the genes used to cluster the samples;
2. For each sample, produce a hierarchical clustering;
3. In a `(n_samples, n_samples)`-shaped matrix, store the number of times a sample pair appears together in a bootstrapped clustering.
4. Perform a hierarchical clustering on the resulting matrix.

This identifies groups of samples that frequently occur together in clusterings, regardless of the genes chosen.
Thus, these samples can be considered to robustly cluster together.

*Hint: use `np.random.choice` with `replacement=True` to create bootstrap samples of row indices.*

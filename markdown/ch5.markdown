```python
%matplotlib inline
# Set up plotting
```

# Contingency tables using sparse coordinate matrices

*Code by Juan Nunez-Iglesias.  
Suggested by Andreas Mueller.*

Many real-world matrices are *sparse*, which means that most of their values are zero.

(Examples.)

Using numpy arrays for these problems wastes a lot of time and energy multiplying many, many values by 0.
Instead, we can use SciPy's `sparse` module to solve these efficiently, examining only non-zero values.

In addition to helping solve these "canonical" sparse matrix problems, `sparse` can be used to problems that were not obviously related to sparse matrices.

One such problem is the comparison of image segmentations.
(Review chapter 3 for a definition of segmentation.)

But let's start simple and work our way up to segmentations.

Suppose you just started working as a data scientist at email startup Spam-o-matic.
You are tasked with building a detector for spam email.
You encode the detector outcome as a numeric value, 0 for not spam and 1 for spam.

If you have a training set of 10 emails to classify, you end up with a vector of *predictions*:

```python
import numpy as np
pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
```

You can check how well you've done by comparing it to a vector of *ground truth*, classifications obtained by inspecting each message by hand.

```python
gt = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
```

Now, classification is hard for computers, so the values in `pred` and `gt` don't match up exactly.
At positions where `pred` is 0 and `gt` is 0, the prediction has correctly identified a message as non-spam.
This is called a *true negative*.
Conversely, at positions where both values are 1, the predictor has correctly identified a spam message, and found a *true positive*.

Then there are two kinds of errors.
If we let a spam message (where `gt` is 1) through to the user's inbox (`pred` is 0), we've made a *false negative* error.
If we predict a legitimate message (`gt` is 0) to be spam (`pred` is 1), we've made a *false positive* prediction.
(An email from the director of my scientific institute once landed in my spam folder. The reason? His announcement of a postdoc talk competition started with "You could win $500!")

If we want to measure how well we are doing, we have to count the above kinds of errors using a *contingency matrix*.
(This is also sometimes called a confusion matrix. The name is apt.)
For this, we place the prediction labels along the rows and the ground truth ones along the columns, and count the number of times they correspond.
So, for example, since there are 4 true positives (where `pred` and `gt` are both 1), the matrix will have a value of 3 at position (1, 1).

Generally:

$$
C_{i, j} = \sum_k{\mathbb{I}\(p_k = i\) \mathbb{I}\(g_k = j\)}
$$

Here's an inefficient way of building the above:

```python
def confusion_matrix(pred, gt):
    cont = np.zeros((2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            cont[i, j] = np.sum((pred == i) & (gt == j))
    return cont
```

We can check that this gives use the right counts:

```python
confusion_matrix(pred, gt)
```

**Question:** Why did we call this inefficient?

**Exercise:** Write an alternative way of computing the confusion matrix that only makes a single pass through `pred` and `gt`.

```python
def confusion_matrix1(pred, gt):
    cont = np.zeros((2, 2))
    # your code goes here
    return cont
```

We can make this example a bit more general:
Instead of classifying spam and non-spam, we can classify spam, newsletters,
sales and promotions, mailing lists, and personal email.
That's 5 categories, which we'll label 0 to 4.
The confusion matrix will now be 5-by-5, with matches counted on the diagonal,
and errors counted on the off-diagonal entries.

The definition of the `confusion_matrix` function, above, doesn't extend well
to this larger matrix, because now we must have *twenty-five* passes though the
results and ground truth arrays.
This problem only grows as we add more email categories, such as social media
notifications.

**Exercise:** Write a function to compute the confusion matrix in one pass, as
above, but instead of assuming two categories, infer the number of categories
from the input.

```python
def general_confusion_matrix(pred, gt):
    n_classes = None  # replace `None` with something useful
    # your code goes here
    return cont
```

Your one-pass solution will scale well with the number of classes, but, because
the for-loop runs in the Python interpreter, it will be slow when you have a
large number of documents.
Also, because some classes are easier to mistake for one another, the matrix
will be *sparse*, with many 0 entries.
Indeed, as the number of classes increases, dedicating lots of memory space to
the 0 entries of the contingency matrix is increasingly wasteful.
Instead, we can use the `sparse` module of SciPy, which contains objects to
efficiently represent sparse matrices.

## scipy.sparse data formats

We covered the internal data format of NumPy arrays in Chapter 1.
I hope you agree that it's a fairly intuitive, and, in some sense, inevitable
format to hold n-dimensional array data.
For sparse matrices, there are actually a wide array of possible formats, and
the "right" format depends on the problem you want to solve.

Perhaps the most intuitive is the coordinate, or COO, format.
This uses three 1D arrays to represent a 2D matrix $A$.
Each of these arrays has length equal to the number of nonzero values in $A$,
and together they list (i, j, value) coordinates of every entry that is not
equal to 0.

- the `i` and `j` arrays, which together specify the location of each non-zero
  entry.
- the `data` array, which specifies the *value* at each location.

Every part of the matrix that is not represented by the `(i, j)` pairs is
considered to be 0.

So, to represent the matrix:

```python
s = np.array([[ 4,  0, 3],
              [ 0, 32, 0]], dtype=float)
```

We can do the following:

```python
from scipy import sparse

data = np.array([4, 3, 32], dtype=float)
i = np.array([0, 0, 1])
j = np.array([0, 2, 1])

scoo = sparse.coo_matrix((data, (i, j)))
```

The `.todense()` method of every sparse format in `scipy.sparse` returns a
numpy array representation of the sparse data.
We can use this to check that we created `scoo` correctly:

```python
scoo.todense()
```

**Exercise**: write out the COO representation of the following matrix:

```python
s2 = np.array([[0, 0, 6, 0, 0],
               [1, 2, 0, 4, 5],
               [0, 1, 0, 0, 0],
               [9, 0, 0, 0, 0],
               [0, 0, 0, 6, 7]])
```

Unfortunately, although the COO format is intuitive, it's not very optimized to
use the minimum amount of memory, or to traverse the array as quickly as
possible during computations.
(Remember from Chapter 1, *data locality* is very important to efficient
computation!)
However, you can look at your COO representation above to help you identify
redundant information:
Notice all those repeated `1`s?

If we use COO to enumerate the nonzero entries row-by-row, rather than in
arbitrary order (which the format allows), we end up with many consecutive,
repeated values in the `i` array.
These can be compressed by indicating the *indices* in `j` where the next row
starts, rather than repeatedly writing the row index.
This is the basis for the *compressed sparse row* or *CSR* format.

Let's work through the example above.
In CSR format, the `j` and `data` arrays are unchanged (but `j` is renamed to
`indices`).
However, the `i` array, instead of indicating the rows, indicates *where* in
`j` each row begins, and is renamed to `indptr`, for "index pointer".

So, let's look at `i` and `j` in COO format, ignoring `data`:

```python
i = [0, 1, 1, 1, 1, 2, 3, 4, 4]
j = [2, 0, 1, 3, 4, 1, 0, 3, 4]
```

Each new row begins at the index where `i` changes.
The 0th row starts at index 0, and the 1st row starts at index 1, but the 2nd
row starts where "2" first appears in `i`, at index 5.
Then, the indices increase by 1 for rows 3 and 4, to 6 and 7.
The final index, indicating the end of the matrix, is the total number of
nonzero values (9).
So:

```python
indptr = [0, 1, 5, 6, 7, 9]
```

Let's use these hand-computed arrays to build a CSR matrix in SciPy.
We can check our work by comparing the `.todense()` output from our COO and
CSR representations to the numpy array `s2` that we defined earlier.

```python
data = np.array([6, 1, 2, 4, 5, 1, 9, 6, 7])

coo = sparse.coo_matrix((data, (i, j)))
csr = sparse.csr_matrix((data, j, indptr))

print('The COO and CSR arrays are equal: ',
      np.all(coo.todense(), csr.todense()))
print('The CSR and NumPy arrays are equal: ',
      np.all(s2, csr.todense()))
```

[This is useful in a very wide array of scientific problems.]

## Applications of sparse matrices

[@stefanv's section]

## Back to contingency matrices

[Extend to *unknown number* of classes]

so, because the COO format (a) only stores a `rows` array, a `columns` array, and a `values` array, and (b) sums the values whenever the same (row column) pair appears twice, we are already done, just by making `rows = pred`, `columns = gt`, and `values = np.ones(pred.size)`!

```python
from scipy import sparse

def confusion_matrix(pred, gt):
    cont = sparse.coo_matrix((np.ones(pred.size), (pred, gt)))
    return cont
```

To look at a small one, we simply use the `.todense()` method, which returns the numpy array corresponding to that matrix:

```python
cont = confusion_matrix(pred, gt)
print(cont)
```

```python
print(cont.todense())
```

# Contingency matrices in segmentation

You can think of the segmentation of an image in the same way as the classification problem above:
The segment label at each *pixel* is a *prediction* about which *class* the pixel belongs to.
And numpy arrays allow us to do this transparently, because their `.ravel()` method returns a 1D view of the underlying data.

As an example, here's a segmentation of a tiny 3 by 3 image:

```python
seg = np.array([[1, 1, 2],
                [1, 2, 2],
                [3, 3, 3]], dtype=int)
```

Here’s the ground truth, what some person said was the correct way to segment this image:

```python
gts = np.array([[1, 1, 1],
                [1, 1, 1],
                [2, 2, 2]], dtype=int)
```

We can think of these two as classifications, just like before:

```python
print(seg.ravel())
print(gts.ravel())
```

Then, like above, the contingency matrix is given by:

```python
cont = sparse.coo_matrix((np.ones(seg.size),
                          (seg.ravel(), gt.ravel())))
print(cont)
```

Segmentation is a hard problem, so it's important to measure how well a segmentation algorithm is doing, by comparing its output to a "ground truth" segmentation that is manually produced by a human.

But, even this comparison is not an easy task.
How do we define how "close" an automated segmentation is to a ground truth?
We'll illustrate one method, the *variation of information* or VI (Meila, 2005).
This is defined as the answer to the following question: on average, for a random pixel, if we are given its segment ID in one segmentation, how much more *information* do we need to determine its ID in the other?
Mathematically, to compare segmentations $A$ and $B$, we define $VI$:

$$
VI = H(A | B) + H(B | A)
$$
where $H(a|b)$ is the conditional entropy of $a$ given $b$:

$$
H(A | B) = \sum_{y \in B}{p(x)H(A | B = y)}
$$

and:

$$
H(A | B=y) = \sum_{x \in A}{\frac{p(xy)}{p(y)}\log_2\(\frac{p(xy)}{p(y)}\)}
$$

Two segmentations of the same image are *label arrays* of the same shape as the image (and therefore, same shape as each other).
The definitions above use just $p(xy)$, the joint probability of labels, and $p(x)$ and $p(y)$, the *marginal* probabilities of labels.
These are merely the fractions of pixels (or voxels, for 3D images) having labels $x$ in $A$ and $y$ in $B$.
To compute these, we just need to count the number of times labels appear together (in the same pixel) in either segmentation.
It turns out that this can be done *easily* by the constructor of *scipy.sparse.coo_matrix*.

```python
def invert_nonzero(mat):
    mat_inv = mat.copy()
    nz = mat.data.nonzero()
    mat_inv.data[nz] = 1 / mat_inv[nz]
    return mat_inv


def xlogx(mat):
    matlog = mat.copy()
    nz = mat.data.nonzero()
    matlog.data[nz] = mat.data[nz] * np.log2(mat.data[nz])
    return matlog


def vi(x, y):
    pxy = sparse.coo_matrix((np.ones(x.size), (x.ravel(), y.ravel())),
                            dtype=float).tocsr()
    pxy.data /= np.sum(pxy.data)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_inv = sparse.diags(invert_nonzero(px), [0])
    py_inv = sparse.diags(invert_nonzero(py), [0])
    hygx = -(px * xlogx(py_inv.dot(pxy)).sum(axis=0)).sum()
    hxgy = -(py * xlogx(pxy.dot(px)).sum(axis=1)).sum()
    return hygx + hxgy
```

[Ed note: Tiger image and segmentation licensed for "non-commercial research and educational purposes"?.
May need to ask permission to use in the book. See: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/]

```python
# VI tiger example from Ch3

from skimage import io
from matplotlib import pyplot as plt

url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg'
tiger = io.imread(url)

plt.imshow(tiger);
```

```python
from scipy import ndimage as nd
from skimage import color

human_seg_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/human/normal/outline/color/1122/108073.jpg'
boundaries = io.imread(human_seg_url)
io.imshow(boundaries);
```

```python
human_seg = nd.label(boundaries > 100)[0]
io.imshow(color.label2rgb(human_seg, tiger));
```

```python
# Draw a region adjacency graph (RAG) - all code from Ch3
import networkx as nx
import numpy as np
from scipy import ndimage as nd
from skimage.future import graph

def add_edge_filter(values, graph):
    current = values[0]
    neighbors = values[1:]
    for neighbor in neighbors:
        graph.add_edge(current, neighbor)
    return 0. # generic_filter requires a return value, which we ignore!

def build_rag(labels, image):
    g = nx.Graph()
    footprint = nd.generate_binary_structure(labels.ndim, connectivity=1)
    for j in range(labels.ndim):
        fp = np.swapaxes(footprint, j, 0)
        fp[0, ...] = 0  # zero out top of footprint on each axis
    _ = nd.generic_filter(labels, add_edge_filter, footprint=footprint,
                          mode='nearest', extra_arguments=(g,))
    for n in g:
        g.node[n]['total color'] = np.zeros(3, np.double)
        g.node[n]['pixel count'] = 0
    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color'] += image[index]
        g.node[n]['pixel count'] += 1
    return g

def threshold_graph(g, t):
    to_remove = ((u, v) for (u, v, d) in g.edges(data=True)
                 if d['weight'] > t)
    g.remove_edges_from(to_remove)
```

```python
# Baseline segmentation
from skimage import segmentation
seg = segmentation.slic(tiger, n_segments=30, compactness=40.0,
                        enforce_connectivity=True, sigma=3)
io.imshow(color.label2rgb(seg, tiger));
```

```python
def RAG_segmentation(base_seg, image, threshold=80):
    g = build_rag(base_seg, image)
    for n in g:
        node = g.node[n]
        node['mean'] = node['total color'] / node['pixel count']
    for u, v in g.edges_iter():
        d = g.node[u]['mean'] - g.node[v]['mean']
        g[u][v]['weight'] = np.linalg.norm(d)

    threshold_graph(g, threshold)

    map_array = np.zeros(np.max(seg) + 1, int)
    for i, segment in enumerate(nx.connected_components(g)):
        for initial in segment:
            map_array[initial] = i
    segmented = map_array[seg]
    return(segmented)
```

```python
# workaround version of vi while the version for this chapter is being fixed
from gala import evaluate
vi = evaluate.vi
```

Try a few thresholds

```python
# Ground truth: human_seg
# Automated segmentation: auto_seg
auto_seg = RAG_segmentation(seg, tiger, threshold=40)
plt.imshow(color.label2rgb(auto_seg, tiger));

vi(auto_seg, human_seg)
```

```python
auto_seg = RAG_segmentation(seg, tiger, threshold=80)
plt.imshow(color.label2rgb(auto_seg, tiger));

vi(auto_seg, human_seg, ignore_x=[], ignore_y=[])
```

```python
# Try many thresholds
def vi_at_threshold(seg, tiger, human_seg, threshold):
    auto_seg = RAG_segmentation(seg, tiger, threshold)
    return(vi(auto_seg, human_seg, ignore_x=[], ignore_y=[]))

thresholds = range(0,110,10)
vi_per_threshold = [vi_at_threshold(seg, tiger, human_seg, threshold) for threshold in thresholds]
```

```python
plt.plot(thresholds, vi_per_threshold);
```

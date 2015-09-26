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

Suppose you just started working as a data scientist at email startup Spamomatic.
You are tasked with building a detector for spam email.
You encode the detector outcome as a numeric value, 0 for not spam and 1 for spam.

If you have a training set of 10 emails to classify, you end up with a vector of *predictions*:

```python
pred = np.array([[0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
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

**Question:** Why did we call this inefficient?

**Exercise:** Write an alternative way of computing the confusion matrix that only makes a single pass through `pred` and `gt`.

```python
def confusion_matrix1(pred, gt):
    cont = np.zeros((2, 2))
    # your code goes here
    return cont
```

[Extend the above to 3 classes instead of 2]

[Extend it to *unknown number* of classes]

[motivate `sparse` and `sparse.coo_matrix` with above]


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

# Contingency tables using coordinate matrices

*Code by Juan Nunez-Iglesias.  
Suggested by Andreas Mueller.*

Many real-world matrices are *sparse*, which means that most of their values are zero.

(Examples.)

Using numpy arrays for these problems wastes a lot of time and energy multiplying many, many values by 0.
Instead, we can use SciPy's `sparse` module to solve these efficiently, examining only non-zero values.

In addition to helping solve these "canonical" sparse matrix problems, `sparse` can be used to problems that were not obviously related to sparse matrices.

One such problem is the comparison of image segmentations.
(Review chapter 3 for a definition of segmentation.)
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
    px_inv = invert_nonzero(px)
    py_inv = invert_nonzero(py)
    hygx = - (px * xlogx(py_inv.dot(pxy)).sum(axis=0)).sum()
    hxgy = - (py * xlogx(pxy.dot(px)).sum(axis=1)).sum()
    return hygx + hxgy
```

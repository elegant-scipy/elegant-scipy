# Linear algebra in SciPy

```python
%matplotlib inline
```

Just like Chapter 4, which dealt with the Fast Fourier Transform, this chapter
will feature an elegant *method*. We
want to highlight the linear algebra packages available in SciPy, which form
the basis of much scientific computing.

A chapter in a programming book is not really the right place to learn about
linear algebra itself, so we assume familiarity with linear algebra concepts.
At a minimum, you should know that linear algebra involves vectors (ordered
collections of numbers) and their transformations by multiplying them with
matrices (collections of vectors). If all of this sounded like gibberish to
you, you should probably pick up an introductory linear algebra textbook before
reading this. Introductory is all you need though — we hope to convey the power
of linear algebra while keeping the operations relatively simple!

As an aside, we will break Python notation convention in order to match linear
algebra conventions: in Python, variables names should usually begin with a
lower case letter. However, in linear algebra, matrices are usually denoted by
a capital letter, while vectors and scalar values are lowercase. Since we're
going to be dealing with quite a few matrices and vectors, it helps to keep
them straight to follow the linear algebra convention. Therefore, variables
that represent matrices will start with a capital letter, while vectors and
numbers will start with lowercase.

## Laplacian matrix of a graph

We discussed graphs in chapter 3, where we represented image regions as
nodes, connected by edges between them. But we used a rather simple method of
analysis: we *thresholded* the graph, removing all edges above some value.
Thresholding works in simple cases, but can easily fail, because all you need
is one noisy edge to fall on the wrong side of the threshold for the approach
to fail.

In this chapter, we will explore some alternative approaches to graph analysis,
based on linear algebra. It turns out that we can think of a graph G as an
*adjacency matrix*, in which we number the nodes of the graph from $0$ to
$n-1$, and place a 1 in row $i$, column $j$ of the matrix whenever there
is an edge from node $i$ to node $j$. In other words, if we call the adjacency
matrix $A$, then $A_{i, j} = 1$ if and only if the link
(i, j) is in G. We can then use linear algebra techniques to study this matrix,
with often striking results.

The degree of a node is the number of edges touching it. For example, if a node
is connected to five other nodes in a graph, its degree is 5. (Later, we
will differentiate between out-degree and in-degree, when edges have a "from"
and "to".)

The *Laplacian* matrix of a graph (just "the Laplacian" for short) is defined
as the *degree matrix*, $D$, which
contains the degree of each node along the diagonal and zero everywhere else,
minus the adjacency matrix $A$:

$$
L = D - A
$$

We definitely can't fit all of the linear algebra theory needed to understand
the properties of this matrix, but suffice it to say: it has some *great*
properties. We will exploit a couple in the following paragraphs.

For example, a common problem in network analysis is visualization. How do you
draw nodes and links in such a way that you don't get a complete mess such as
this one?

![network hairball](https://upload.wikimedia.org/wikipedia/commons/9/90/Visualization_of_wiki_structure_using_prefuse_visualization_package.png)
**Graph created by Chris Davis. [CC-BY-SA-3.0](https://commons.wikimedia.org/wiki/GNU_Free_Documentation_License).**
**[Ed note: my reading is that we can use this figure as long as we include the
copyright notice and don't put DRM on the books, which O'Reilly doesn't do
anyway.]**

One way is to put nodes that share many links close together, and it turns out
that this can be done by using the second-smallest eigenvalue of the Laplacian
matrix, and its corresponding eigenvector, which is so important it has its
own name: the
[Fiedler vector](https://en.wikipedia.org/wiki/Algebraic_connectivity#The_Fiedler_vector).

As a quick aside, an eigenvector $v$ of a matrix $M$ is a vector that
satisfies the property $Mv = \lambda v$ for some number $\lambda$,
known as the eigenvalue.  In other words, $v$ is a special vector in
relation to $M$ because $Mv$ simply scales the vector, without
changing its direction.[^eigv_example]

[^eigv_example]: As an example, consider a 3x3 rotation matrix $R$
                 that, when multiplied by any 3-dimensional vector
                 $p$, rotates it $30^\circ$ degrees around the z-axis.
                 $R$ will rotate all vectors except for those that lie
                 *on* the z-axis.  For those, we'll see no effect, or
                 $Rp = p$, i.e. $Rp = \lambda p$ with eigenvalue
                 $\lambda = 1$.

<!-- exercise begin -->

**Exercise:** Consider the rotation matrix

$$
R = \begin{bmatrix}
  \cos \theta &  -\sin \theta & 0 \\[3pt]
  \sin \theta & \cos \theta & 0\\[3pt]
  0 & 0 & 1\\
\end{bmatrix}
$$

When R is multiplied with a 3-dimensional column-vector $p =
\left[ x\, y\, z \right]^T$, the resulting vector $R p$ is rotated
by $\theta$ degrees around the z-axis.

1. For $\theta = 45^\circ$, verify (by testing on a few arbitrary
   vectors) that R rotates these vectors around the z axis.

2. Now, verify that multiplying by $R$ leaves the vector
   $\left[ 0\, 0\, 1\right]^T$ unchanged.  In other words, $R p = 1
   p$, which means $p$ is an eigenvector of R with eigenvalue 1.

<!-- solution begin -->

**Solution:**

```python
import numpy as np

theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,              0,             1]])

print("R @ x-axis:", R @ [1, 0, 0])
print("R @ y-axis:", R @ [0, 1, 0])
print("R @ z-axis:", R @ [0, 0, 1])

```

R rotates both the x and y axes, but not the z-axis.

----

<!-- solution end -->

<!-- exercise end -->

The eigenvectors have numerous useful--sometimes almost magical!--properties.

Let's use a minimal network to illustrate this. We start by creating the
adjacency matrix:

```python
import numpy as np
A = np.array([[0, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]], dtype=float)
```

We can use NetworkX to draw this network:

```python
import networkx as nx
g = nx.from_numpy_matrix(A)
nx.draw_spring(g, with_labels=True, node_color='white')
```

You can see that the nodes fall naturally into two groups, 0, 1, 2 and 3, 4, 5.
Can the Fiedler vector tell us this? First, we must compute the degree matrix
and the Laplacian. We first get the degrees by summing along either axis of A.
(Either axis works because A is symmetric.)

```python
d = np.sum(A, axis=0)
print(d)
```

We then put those degrees into a diagonal matrix of the same shape
as A, the *degree matrix*. We can use `scipy.sparse.diags` to do this:

```python
from scipy import sparse
D = sparse.diags(d).toarray()
print(D)
```

Finally, we get the Laplacian from the definition:

```python
L = D - A
print(L)
```

Because $L$ is symmetric, we can use the `np.linalg.eigh` function to compute
the eigenvalues and eigenvectors:

```python
eigvals, Eigvecs = np.linalg.eigh(L)
```

You can verify that the values returned satisfy the definition of eigenvalues
and eigenvectors. For example, the third eigenvalue is 3:

```python
eigvals[2]
```

And we can check that multiplying the matrix $L$ by the second eigenvector does
indeed multiply the vector by 3:

```python
v2 = Eigvecs[:, 2]

print(v2)
print(L @ v2)
```

As mentioned above, the Fiedler vector is the vector corresponding to the
second-smallest eigenvalue of $L$. Plotting the eigenvalues tells us which one
is the second-smallest:

```python
from matplotlib import pyplot as plt
print(eigvals)
plt.stem(eigvals)
```

It's the second eigenvalue. The Fiedler vector is thus the second eigenvector.

```python
f = Eigvecs[:, 1]
plt.stem(f)
```

It's pretty remarkable: by looking at the *sign* of the Fiedler vector, we can
separate the nodes into the two groups we identified in the drawing!

```python
colors = ['orange' if eigval > 0 else 'gray' for eigval in f]
nx.draw_spring(g, with_labels=True, node_color=colors)
```

Let's demonstrate this in a real-world example by laying out the brain cells in a worm, as shown in
[Figure 2](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066)
from the
[Varshney *et al* paper](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066)
that we introduced in Chapter 3. (Information on
how to do this is in the
[supplementary material](http://journals.plos.org/ploscompbiol/article/asset?unique&id=info:doi/10.1371/journal.pcbi.1001066.s001)
for the paper.) To obtain their
layout of the worm brain neurons, they used a related matrix, the
*degree-normalized Laplacian*.

Because the order of the neurons is important in this analysis, we will use a
preprocessed dataset, rather than clutter this chapter with data cleaning. We
got the original data from Lars Varshney's
[website](http://www.ifp.illinois.edu/~varshney/elegans),
and the processed data is in our `data/` directory.

First, let's load the data. There are four components:
- the network of chemical synapses, through which a *pre-synaptic neuron*
  sends a chemical signal to a *post-synaptic* neuron,
- the gap junction network, which contains direct electrical contacts between
  neurons),
- the neuron IDs (names), and
- the three neuron types:
  - *sensory neurons*, those that detect signals coming from the outside world,
    encoded as 0;
  - *motor neurons*, those that activate muscles, enabling the worm to move,
    encoded as 2; and
  - *interneurons*, the neurons in between, which enable complex signal processing
    to occur between sensory neurons and motor neurons, encoded as 1.

```python
import numpy as np
chem = np.load('data/chem-network.npy')
gap = np.load('data/gap-network.npy')
neuron_ids = np.load('data/neurons.npy')
neuron_types = np.load('data/neuron-types.npy')
```

We then simplify the network, adding the two kinds of connections together,
and removing the directionality of the network by taking the average of
in-connections and out-connections of neurons. This seems a bit like cheating
but, since we are only looking for the *layout* of the neurons on a graph, we
only care about *whether* neurons are connected, not in which direction.
We are going to call the resulting matrix the *connectivity* matrix, $C$, which
is just a different kind of adjacency matrix.

```python
A = chem + gap
C = (A + A.T) / 2
```

To get the Laplacian matrix $L$, we need the degree matrix $D$, which contains
the degree of node i at position [i, i], and zeros everywhere else.

```python
n = C.shape[0]
D = np.zeros((n, n), dtype=np.float)
diag = (np.arange(n), np.arange(n))
D[diag] = np.sum(C, axis=0)
L = D - C
```

The vertical coordinates in Fig 2 are given by arranging nodes such that, on
average, neurons are as close as possible to "just above" their downstream
neighbors. Varshney _et al_ call this measure "processing depth," and it's
obtained by solving a linear equation involving the Laplacian. We use
`scipy.linalg.pinv`, the
[pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse),
to solve it:

```python
from scipy import linalg
b = np.sum(C * np.sign(A - A.T), axis=1)
z = linalg.pinv(L) @ b
```

(Note the use of the `@` symbol, which was introduced in Python 3.5 to denote
matrix multiplication. As we noted in the preface and in Chapter 5, in previous
versions of Python, you would need to use the function `np.dot`.)

In order to obtain the degree-normalized Laplacian, Q, we need the inverse
square root of the D matrix:

```python
Dinv2 = np.zeros((n, n))
Dinv2[diag] = D[diag] ** (-.5)
Q = Dinv2 @ L @ Dinv2
```

Finally, we are able to extract the x coordinates of the neurons to ensure that
highly-connected neurons remain close: the eigenvector of Q corresponding to
its second-smallest eigenvalue, normalized by the degrees:

```python
eigvals, eigvecs = linalg.eig(Q)
```

Note from the documentation of `numpy.linalg.eig`:

> "The eigenvalues are not necessarily ordered."

Although the documentation in SciPy's `eig` lacks this warning
(disappointingly, we must add), it remains true in this case. We must therefore
sort the eigenvalues and the corresponding eigenvector columns ourselves:

```python
smallest_first = np.argsort(eigvals)
eigvals = eigvals[smallest_first]
eigvecs = eigvecs[:, smallest_first]
```

Now we can find the eigenvector we need to compute the affinity coordinates:

```python
x = Dinv2 @ eigvecs[:, 1]
```

(The reasons for using this vector are too long to explain here, but appear in
the paper's supplementary material, linked above.)

Now it's just a matter of drawing the nodes and the links. We color them
according to the type stored in `neuron_types`:

```python
from matplotlib import pyplot as plt
from matplotlib import colors

def plot_connectome(neuron_x, neuron_y, links, labels, types):
    colormap = colors.ListedColormap([[ 0.   ,  0.447,  0.698],
                                      [ 0.   ,  0.62 ,  0.451],
                                      [ 0.835,  0.369,  0.   ]])
    # plot neuron locations:
    points = plt.scatter(neuron_x, neuron_y, c=types, cmap=colormap,
                         edgecolors='face', zorder=1)

    # add text labels:
    for x, y, label in zip(neuron_x, neuron_y, labels):
        plt.text(x, y, '  ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=5, zorder=2)

    # plot links
    pre, post = np.nonzero(links)
    for src, dst in zip(pre, post):
        plt.plot(neuron_x[[src, dst]], neuron_y[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)

    plt.show()
```

```python
plt.figure(figsize=(16, 9))
plot_connectome(x, z, C, neuron_ids, neuron_types)
```

There you are: a worm brain!
As discussed in the original paper, you can see the top-down processing from
sensory neurons to motor neurons through a network of interneurons. You can
also see two distinct groups of motor neurons: these correspond to the neck
(left) and body (right) body segments of the worm.

<!-- exercise begin -->

**Exercise:** How do you modify the above code to show the affinity view in
Figure 2B from the paper?

<!-- solution begin -->
**Solution:** In the affinity view, instead of using the processing depth on the y-axis,
we use the normalized third eigenvector of Q, just like we did with x:

```python
y = Dinv2 @ eigvecs[:, 2]
plt.figure(figsize=(16, 9))
plot_connectome(x, y, C, neuron_ids, neuron_types)
```
<!-- solution end -->

<!-- exercise end -->

<!-- exercise begin -->

### Challenge: linear algebra with sparse matrices

The above code uses numpy arrays to hold the matrix and perform
the necessary computations. Because we are using a small graph of fewer than 300
nodes, this is feasible. However, for larger graphs, it would fail.

In what follows, we will analyze the dependency graph for packages in the
Python Package Index, or PyPI, which contains over 75 thousand packages. To
hold the Laplacian matrix for this graph would take up
$8 \left(75 \times 10^3\right)^2 = 45 \times 10^9$ bytes, or 45GB,
of RAM. If you add to that the adjacency, symmetric adjacency, pseudoinverse,
and, say, two temporary matrices used during calculations, you climb up to
270GB, beyond the reach of most desktop computers.

"Ha!", some of you might be thinking. "Ha! My desktop has 512GB of RAM! It would
make short work of this so-called 'large' graph!"

Perhaps. But we will also be analysing the Association for Computing Machinery
(ACM) citation graph, a network of over two million scholarly works and
references. *That* Laplacian would take up 32 terabytes of RAM.

However, we know that the dependency and reference graphs are *sparse*:
packages usually depend on just a few other packages, not on the whole of PyPI.
And papers and books usually only reference a few others, too. So we can hold
the above matrices using the sparse data structures from `scipy.sparse` (see
Chapter 5), and use the linear algebra functions in `scipy.sparse.linalg` to
compute the values we need.

Try to explore the documentation in `scipy.sparse.linalg` to come up with a
sparse version of the above computation.

Hint: the pseudoinverse of a sparse matrix is, in general, not sparse, so you
can't use it here. Similarly, you can't get all the eigenvectors of a sparse
matrix, because they would together make up a dense matrix.

You'll find parts of the solution below (and of course in the solutions
chapter), but we highly recommend that you try it out on your own.

<!-- solution begin -->

### Challenge accepted

For the purposes of this challenge, we are going to use the small connectome
above, because it's easier to visualise what is going on. In later parts of the
challenge we'll use these techniques to analyze larger networks.

First, we start with the adjacency matrix, A, in a sparse matrix format, in
this case, CSR, which is the most common format for linear algebra. We'll
append `s` to the names of all the matrices to indicate that they are sparse.

```python
from scipy import sparse
import scipy.sparse.linalg

As = sparse.csr_matrix(A)
```

We can create our connectivity matrix in much the same way:

```python
Cs = (As + As.T) / 2
```

In order to get the degrees matrix, we can use the "diags" sparse format, which
stores diagonal and off-diagonal matrices.

```python
degrees = np.ravel(Cs.sum(axis=0))
Ds = sparse.diags(degrees, 0)
```

Getting the Laplacian is straightforward:

```python
Ls = Ds - Cs
```

Now we want to get the processing depth. Remember that getting the
pseudo-inverse of the Laplacian matrix is out of the question, because it will
be a dense matrix (the inverse of a sparse matrix is not generally sparse
itself). However, we were actually using the pseudo-inverse to compute a vector
$z$ that would satisfy $L z = b$, where
$b = C \odot \textrm{sign}\left(A - A^T\right) \mathbb{1}$.
(You can see this in the supplementary material for Varshney *et al*.) With
dense matrices, we can simply use $z = L^+b$. With sparse ones, though, we can
use one of the *solvers* in `sparse.linalg.isolve` to get the `z` vector after
providing `L` and `b`, no inversion required!

```python
b = Cs.multiply((As - As.T).sign()).sum(axis=1)
z, error = sparse.linalg.isolve.cg(Ls, b, maxiter=10000)
```

Finally, we must find the eigenvectors of $Q$, the degree-normalized Laplacian,
corresponding to its second and third smallest eigenvalues.

You might recall from Chapter 5 that the numerical data in sparse matrices is
in the `.data` attribute. We use that to invert the degrees matrix:

```python
Dsinv2 = Ds.copy()
Dsinv2.data = Ds.data ** (-0.5)
```

Finally, we use SciPy's sparse linear algebra functions to find the desired
eigenvectors. The $Q$ matrix is symmetric, so we can use the `eigsh` function,
specialized for symmetric matrices, to compute them. We use the `which` keyword
argument to specify that we want the eigenvectors corresponding to the smallest
eigenvalues, and `k` to specify that we need the 3 smallest:

```python

Qs = Dsinv2 @ Ls @ Dsinv2
eigvals, eigvecs = sparse.linalg.eigsh(Qs, k=3, which='SM')
sorted_indices = np.argsort(eigvals)
eigvecs = eigvecs[:, sorted_indices]
```

Finally, we normalize the eigenvectors to get the x and y coordinates:

```python
_dsinv, x, y = (Dsinv2 @ eigvecs).T
```

(Note that the eigenvector corresponding to the smallest eigenvalue is always a
vector of all ones, which we're not interested in.)
We can now reproduce the above plots!

```python
plt.figure(figsize=(16, 9))
plot_connectome(x, z, C, neuron_ids, neuron_types)

plt.figure(figsize=(16, 9))
plot_connectome(x, y, C, neuron_ids, neuron_types)
```

Note that eigenvectors are defined only up to a (possibly negative)
multiplicative constant, so the plots may have ended up reversed! (That is,
left is right, or up is down, or both!)

<!-- solution end -->

<!-- exercise end -->

## Pagerank: linear algebra for reputation and importance

Another application of linear algebra and eigenvectors is Google's Pagerank
algorithm, which is punnily named both for webpages and for one of its
co-founders, Larry Page.

If you're trying to rank webpages by importance, one thing you might look at
is how many other webpages link to it. After all, if everyone is linking to a
particular page, it must be good, right? But the problem is that this metric is
easily gamed: to make your own webpage rise in the rankings, you just have to
create as many other webpages as you can and have them all link to your
original page.

The key insight that drove Google's early success was that important webpages
are not just linked to by many webpages, but also by *other*, *important*
webpages. And how do we know that those other pages are important? Because
they themselves are linked to by important pages. And so on.
As we will see, this recursive definition implies that page
importance can be measured by the eigenvector corresponding to the largest
eigenvalue of the so-called *transition matrix*. This matrix imagines a web
surfer, often named Webster, randomly clicking a link from each webpage he
visits, and then asks, what's the probability that he ends up at any given
page? This probability is called the pagerank.

Since Google's rise, researchers have been applying pagerank to all sorts of
networks. We'll start with an example by Stefano Allesina and Mercedes Pascual,
which they
[published](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000494i)
in PLoS Computational Biology. They thought to apply the method in ecological
*food webs*, networks that link species to those that they eat.

Naively, if you wanted to see how critical a species was for an ecosystem, you
would look at how many species eat it. If it's many, and that species
disappeared, then all its "dependent" species might disappear with it. In
network parlance, you could say that its *in-degree* determines its ecological
importance.

Could pagerank be a better measure of importance for an ecosystem?

Professor Allesina kindly provided us with a few food webs to play around
with. We've saved one of these, from the St Marks National Wildlife Refuge in
Florida, in the Graph Markup Language format. The web was
[described](http://www.sciencedirect.com/science/article/pii/S0304380099000228)
in 1999 by Robert R. Christian and Joseph J. Luczovich. In the dataset, a node
$i$ has a link to node $j$ if species $i$ eats species $j$.

We'll start by loading in the data, which NetworkX knows how to read trivially:

```python
import networkx as nx

stmarks = nx.read_gml('data/stmarks.gml')
```

Next, we get the sparse matrix corresponding to the graph. Because a matrix
only holds numerical information, we need to maintain a separate list of
package names corresponding to the matrix rows/columns:

```python
species = np.array(stmarks.nodes())  # array for multi-indexing
Adj = nx.to_scipy_sparse_matrix(stmarks, dtype=np.float64)
```

From the adjacency matrix, we can derive a *transition probability* matrix,
where every link is replaced by a *probability* of 1 over the number of
outgoing links from that species. In the food web, it might make more sense
to call this a lunch probability matrix.

The total number of species in our matrix is going to be used a lot, so let's
call it $n$:

```python
n = len(species)
```

Next, we need the degrees, and, in particular, the *diagonal matrix* containing
the inverse of the out-degrees of each node on the diagonal:

```python
np.seterr(divide='ignore')  # ignore division-by-zero errors
from scipy import sparse

degrees = np.ravel(Adj.sum(axis=1))
Deginv = sparse.diags(1 / degrees).tocsr()
```

```python
Trans = (Deginv @ Adj).T
```

Normally, the pagerank score would simply be the first eigenvector of the
transition matrix. If we call the transition matrix $M$ and the vector of
pagerank values $r$, we have:

$$
\boldsymbol{r} = M\boldsymbol{r}
$$

But the `np.seterr` call above is a clue that it's not quite
so simple. The pagerank approach only works when the
transition matrix is a *column-stochastic* matrix, in which every
column sums to 1. Additionally, every page must be reachable
from every other page, even if the path to reach it is very long.

In our food web, this causes problems, because the bottom of the food chain,
what the authors call *detritus* (basically sea sludge), doesn't actually *eat*
anything (the Circle of Life notwithstanding), so you can't reach other species
from it.

To deal with this, the pagerank algorithm uses a so-called "damping
factor", usually taken to be 0.85. This means that 85% of the time, the
algorithm follows a link at random, but for the other 15%, it randomly jumps to
any arbitrary page. It's as if every page had a low probability link to every
other page. Or, in our case, it's as if shrimp, on rare occasions, ate sharks.
It might seem non-sensical but bear with us! It is, in fact, the mathematical
representation of the Circle of Life. We'll set it to 0.99, but actually it
doesn't really matter for this analysis: the results are similar for a large
range of possible damping factors.

If we call the damping factor $d$, then the modified pagerank equation is:

$$
\boldsymbol{r} = dM\boldsymbol{r} + \frac{1-d}{n} \boldsymbol{1}
$$

and

$$
(\boldsymbol{I} - dM)\boldsymbol{r} = \frac{1-d}{n} \boldsymbol{1}
$$

We can solve this equation using `scipy.sparse`'s iterative
*biconjugate gradient (bicg)* solver^[bicgstab].  (Note that, if you
had a symmetric matrix, you would use Conjugate Gradients,
``scipy.sparse.linalg.solve.cg``, instead.)

[^bicgstab]: The default formulation for bicg iteration is numerically
    unstable, so we use the stabilized variation, which replaces every
    second iteration with a *generalized minimal residual method*
    (GMRES) iteration.

```python
from scipy.sparse.linalg.isolve import bicgstab as bicg

damping = 0.99

I = sparse.eye(n, format='csc')  # Same sparse format as Trans

pagerank, error = bicg(I - damping * Trans,
                       (1-damping) / n * np.ones(n),
                       maxiter=int(1e4))
print('error code: ', error)
```

As can be seen in the documentation for the `bicg` solver, an error code of 0
indicates that a solution was found! We now have the "foodrank" of the
St. Marks food web!

So how does a species' foodrank compare to the number of other species eating
it?

```python
def pagerank_plot(names, in_degrees, pageranks,
                  annotations=[]):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(in_degrees, pageranks, c=[0.835, 0.369, 0], lw=0)
    labels = []
    for name, indeg, pr in zip(names, in_degrees, pageranks):
        if name in annotations:
            text = ax.text(indeg + 0.1, pr, name)
            labels.append(text)
    ax.set_ylim(0, np.max(pageranks) * 1.1)
    ax.set_xlim(-1, np.max(in_degrees) * 1.1)
    ax.set_ylabel('PageRank')
    ax.set_xlabel('In-degree')

interesting = ['detritus', 'phytoplankton', 'benthic algae', 'micro-epiphytes',
               'microfauna', 'zooplankton', 'predatory shrimps', 'meiofauna', 'gulls']
in_degrees = np.ravel(Adj.sum(axis=0))
pagerank_plot(species, in_degrees, pagerank, annotations=interesting)
```

Having explored the dataset ahead of time, we have pre-labeled some interesting
nodes in the plot. Sea sludge is the most important element both by number of
species feeding on it (15) and by pagerank (>0.003). But the second most
important element is *not* benthic algae, which feeds 13 other species, but
rather phytoplankton, which feeds just 7! That's because other *important*
species feed on it! On the bottom left, we've got sea gulls, who, we can now
confirm, do bugger-all for the ecosystem. Those vicious *predatory shrimps*
(we're not making this up) support the same number of species as phytoplankton,
but they are less essential species, so they end up with a lower foodrank.

Although we won't do it here, Allesina and Pascual go on to model the
ecological impact of species extinction, and indeed find that pagerank
predicts ecological importance better than in-degree.

Before we move on though, we'll note that pagerank can be computed several
different ways. One way, complementary to what we did above, is called the
*power method*, and it's pretty, well, powerful! It stems from the
[Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem),
which states, among other things, that a stochastic matrix has 1 as an
eigenvalue, and that this is its *largest* eigenvalue. (The corresponding
eigenvector is the pagerank vector.) What this means is that, whenever we
multiply *any* vector by $M$, its component pointing towards this major
eigenvector stays the same, while *all other components shrink* by a
multiplicative factor! The consequence is that if we multiply some random
starting vector by $M$ repeatedly, we should eventually get the pagerank
vector.

SciPy makes this very efficient with its sparse matrix module:

```python
def power(Trans, damping=0.85, max_iter=int(1e5)):
    n = Trans.shape[0]
    r0 = np.full(n, 1/n)
    r = r0
    for _iter_num in range(max_iter):
        rnext = damping * Trans @ r + (1 - damping) / n
        if np.allclose(rnext, r):
            print('converged')
            break
        r = rnext
    return r
```

<!-- exercise begin -->

**Exercise:** In the above iteration, note that `Trans` is *not*
column-stochastic, so the vector gets shrunk at each iteration. In order to
make the matrix stochastic, we have to replace every zero-column by a column of
all $1/n$. This is too expensive, but computing the iteration is cheaper. How
can you modify the code above to ensure that $r$ remains a probability vector
throughout?

<!-- solution begin -->

**Solution:** In order to have a stochastic matrix, all columns of the
transition matrix must sum to 1. This is not satisfied when a package doesn't
have any dependencies: that column will consist of all zeroes. Replacing all
those columns by $1/n \boldsymbol{1}$, however, would be expensive.

The key is to realise that *every row* will contribute the *same amount* to the
multiplication of the transition matrix by the current probability vector. That
is to say, adding these columns will add a single value to the result of the
iteration multiplication. What value? $1/n$ times the elements of $r$ that
correspond to a dangling node. This can be expressed as a dot-product of a
vector containing $1/n$ for positions corresponding to dangling nodes, and zero
elswhere, with the vector $r$ for the current iteration.

```python
def power2(Trans, damping=0.85, max_iter=int(1e5)):
    n = Trans.shape[0]
    is_dangling = np.ravel(Trans.sum(axis=0) == 0)
    dangling = np.zeros(n)
    dangling[is_dangling] = 1 / n
    r0 = np.ones(n) / n
    r = r0
    for _ in range(max_iter):
        rnext = (damping * (Trans @ r + dangling @ r) +
                 (1 - damping) / n)
        if np.allclose(rnext, r):
            return rnext
        else:
            r = rnext
    return r
```

Try this out manually for a few iterations. Notice that if you start with a
stochastic vector (a vector whose elements all sum to 1), the next vector will
still be a stochastic vector. Thus, the output pagerank from this function will
be a true probability vector, and the values will represent the probability
that we end up at a particular species when following links in the food chain.

<!-- solution end -->

<!-- exercise end -->


<!-- exercise begin -->

**Exercise:** Verify that these three methods all give the same ranking for the
nodes. `numpy.corrcoef` might be a useful function for this.

<!-- solution begin -->

**Solution:** `np.corrcoef` gives the Pearson correlation coefficient between
all pairs of a list of vectors. This coefficient will be equal to 1 if and only
if two vectors are scalar multiples of each other. Therefore, a correlation
coefficient of 1 is sufficient to show that the above methods produce the same
ranking.

```python
pagerank_power = power(Trans)
pagerank_power2 = power2(Trans)
np.corrcoef([pagerank, pagerank_power, pagerank_power2])
```

<!-- solution end -->

<!-- exercise end -->

<!-- exercise begin -->

**Exercise:** While we were writing this chapter, we started out by computing
pagerank on the graph of Python dependencies. We eventually found that we could
not get nice results with this graph. The correlation between in-degree and
pagerank was much higher than in other datasets, and the few outliers didn't
make much sense to us.

Can you think of three reasons why pagerank might not be the best
measure of importance for the Python dependency graph?

<!-- solution begin -->

**Solution:**

Here's our theories. Yours might be different!

First, the graph of dependencies is fundamentally different to the web. In the
web, important pages reinforce each other: the Wikipedia pages for Newton's
theory of gravitation and Einstein's theory of general relativity link to each
other. Thus, our hypothetical Webster has some probability of bouncing between
them. In contrast, Python dependencies form a directed acyclic graph (DAG).
Whenever Debbie (our hypothetical dependency browser) arrives at a foundational
package such as NumPy, she is instantly transported to a random package,
instead of staying in the field of similar packages.

Second, the DAG of dependencies is shallow: libraries will depend on, for
example, scikit-learn, which depends on scipy, which depends on numpy, and
that's it. Therefore, there is not much room for important packages to link to
other important packages. The hierarchy is flat, and the importance is captured
by the direct dependencies.

Third, scientific programming itself is particularly prone to that flat
hierarchy problem, more so than e.g. web frameworks, because scientists are
doing the programming, rather than professional coders. In particular, a
scientific analysis might end up in a script attached to a paper, or a Jupyter
notebook, rather than another library on PyPI. We hope that this book will
encourage more scientists to write great code that others can reuse, and put it
on PyPI!

<!-- solution end -->

<!-- exercise end -->

## Community detection

We saw a hint in the introduction to this chapter that the Fiedler vector could
be used to detect "communities" in networks, groups of nodes that are tightly
connected to each other but not so much to nodes in other groups.
Mark Newman published a
[seminal paper](http://www.pnas.org/content/103/23/8577.short)
on the topic in 2006, and
[refined it further](http://arxiv.org/abs/1307.7729) in 2013. We'll apply it
to the Python library dependency graph, which will tell us whether Python users
fall into two or more somewhat independent groups.

We've downloaded and preprocessed dependency data from PyPI ahead of
time, available as the file `pypi-dependencies.txt` in the `data/`
folder. The data file consists of a list of library-dependency pairs,
one per line, e.g. ``scipy numpy``. The networkx library that we
started using in Chapter 3 makes it easy to build a graph from these
data. We will use a directed graph since the relationship is
asymmetrical: one library depends on the other, not vice-versa.

```python
import networkx as nx

dependencies = nx.DiGraph()

with open('data/pypi-deps.txt') as lines:
    lib_dep_pairs = (str.split(line) for line in lines)
    dependencies.add_edges_from(lib_dep_pairs)
```

We can then get some statistics about this (incomplete) dataset by using
networkx's built-in functions:

```python
print('Number of packages: ', dependencies.number_of_nodes())
print('Total number of dependencies: ', dependencies.number_of_edges())
```

What is the single most used Python package?

```python
print(max(dependencies.in_degree_iter(),
          key=lambda x: x[1]))
```

We're not going to cover it in this book, but `setuptools` is not a
surprising winner here. It probably belongs in the Python standard
library, in the same category as `os`, `sys`, and others!

Since using setuptools is almost a requirement for being listed in PyPI, we
will remove it from the dataset, given its undue influence
on the shape of the graph.

```python
dependencies.remove_node('setuptools')
```

What's the next most depended-upon package?

```python
print(max(dependencies.in_degree_iter(),
          key=lambda x: x[1]))
```

The requests library is the foundation of a very large fraction of the web
frameworks and web processing libraries.

We can similarly find the top-40 most depended-upon packages:

```python
packages_by_in = sorted(dependencies.in_degree_iter(),
                        key=lambda x: x[1], reverse=True)
for i, p in enumerate(packages_by_in, start=1):
    print(i, '. ', p[0], p[1])
    if i > 40:
        break
```

By this ranking, NumPy ranks 4 and SciPy 27 out of all of PyPI. Not
bad!  Overall, though, one gets the impression that the web community
dominates PyPI. As also mentioned in the preface, this is not
altogether surprising: the scientific Python community is still young
and growing, and web tools are arguably of more generic application.

Let's see whether we can isolate the scientific community.

Because it's unwieldy to draw 90,000 nodes, we are only going to draw a
fraction of PyPI, following the same ideas we used for the worm brain.
Let's look at the top 10,000 packages in PyPI, according to number
of dependencies:

```python
n = 10000
top_names = [p[0] for p in packages_by_in[:n]]
top_subgraph = nx.subgraph(dependencies, top_names)
Dep = nx.to_scipy_sparse_matrix(top_subgraph, nodelist=top_names)
```

As above, we need the connectivity matrix, the symmetric version of the
adjacency matrix:

```python
Conn = (Dep + Dep.T) / 2
```

And the diagonal matrix of its degrees, as well as its inverse-square-root:

```python
degrees = np.ravel(Conn.sum(axis=0))
Deg = sparse.diags(degrees).tocsr()
Dinv2 = sparse.diags(degrees ** (-.5)).tocsr()
```

From this we can generate the Laplacian of the dependency graph:

```python
Lap = Deg - Conn
```

We can then generate an affinity view of these nodes, as shown above for
the worm brain graph. We just need the second and third smallest eigenvectors
of the *affinity matrix*, the degree-normalized version of the Laplacian. We
can use `sparse.linalg.eigsh` to obtain these.

There is a small kink though: because the graph is disconnected, the resulting
eigenvalue problem is ill-defined, and we have to add a bit of the identity
matrix to our affinity matrix.

```python
I = sparse.eye(Lap.shape[0], format='csr')
sigma = 0.5

Affn = Dinv2 @ Lap @ Dinv2
v0 = np.ones(Lap.shape[0])

eigvals, vec = sparse.linalg.eigsh(Affn + sigma*I, k=3, which='SM', v0=v0)

sorted_indices = np.argsort(eigvals)
eigvals = eigvals[sorted_indices]
vec = vec[:, sorted_indices]

_ignored, x, y = (Dinv2 @ vec).T
```

That should give us a nice layout for our Python packages!

```python
plt.figure()
plt.scatter(x, y)
```

That's looking promising! But, obviously, this plot is missing some critical
details! Which packages depend on which? Where are the most important packages?
Perhaps most critically, do particular packages naturally belong together?
In the worm brain, we had prior information about neuron types that helped us
to make sense of the graph. Let's try to infer some similar information about
PyPI packages.

In the same way that the eigen-decomposition of the Laplacian matrix can tell
us about graph layout, it can also tell us about connectivity.

Aim: graph of top Python packages:
- colored by community
- sized by dependencies
- alpha-d by pagerank
- laid out by spectral affinity

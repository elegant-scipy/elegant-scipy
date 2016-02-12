# Linear algebra in SciPy

```python
%matplotlib inline
```

Just like Chapter 4, which dealt with the Fast Fourier Transform, this chapter
will feature an elegant *method*, not so much code, which is simple. We simply
want to highlight the linear algebra packages available in SciPy, which form
the basis of much scientific computing.

## Straight linear algebra problem

e.g. one-layer NN, kernel PCA, or NN from scratch

## Graph Laplacian

We discussed graphs in chapter 3, but used a rather simple method of analysis:
*thresholding* the graph, meaning, removing all the links having weight below
some threshold. It turns out that we can think of a graph G as an *adjacency
matrix* $A$, in which $A_{i, j} = 1$ if and only if the link (i, j) is in G.
We can then use linear algebra techniques to study this matrix, with striking
results.

The *Laplacian* matrix of a graph is defined as the *degree matrix*, $D$, which
contains the degree of each node along the diagonal and 0 everywhere else,
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
matrix.

https://en.wikipedia.org/wiki/Algebraic_connectivity#The_Fiedler_vector

[Example with simple network]

Here, we will demonstrate this by reproducing
[Figure 2](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066)
from the
[Varshney *et al* paper](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066)
on the worm brain that we introduced in Chapter 3. (Information on
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

First, let us load the data. There are four components: the chemical synapse
network, the gap junction network (these are direct electrical contacts between
neurons), the neuron IDs (names), and the three neuron types:
- sensory neurons, those that detect signals coming from the outside world,
  encoded as 0;
- motor neurons, those that activate muscles, enabling the worm to move,
  encoded as 2; and
- interneurons, the neurons in between, which enable complex signal processing
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

```python
A = chem + gap
C = (A + A.T) / 2
```

To get the Laplacian matrix, we need the degree matrix, which contains the
degree of node i at position [i, i], and zeros everywhere else.

```python
n = C.shape[0]
D = np.zeros((n, n), dtype=np.float)
diag = (np.arange(n), np.arange(n))
D[diag] = np.sum(C, axis=0)
L = D - C
```

The vertical coordinates in Fig 2 are given by arranging nodes such that, on
average, neurons are as close as possible to "just above" their downstream
neighbors. Varshney et all call this measure "processing depth," and it's
obtained by solving a linear equation involving the Laplacian. We use
`scipy.linalg.pinv`, the pseudoinverse, to solve it:

```python
from scipy import linalg
b = np.sum(C * np.sign(A - A.T), axis=1)
z = linalg.pinv(L) @ b
```

(Note the use of the `@` symbol, which was introduced in Python 3.5 to denote
matrix multiplication. In previous versions of Python, you would need to use
the function `np.dot`.)

In order to obtain the degree-normalized Laplacian, Q, we need the inverse
square root of the D matrix:

```python
Dinv2 = np.zeros((n, n))
Dinv2[diag] = D[diag] ** (-.5)
Q = Dinv2 @ L @ Dinv2
```

Finally, we are able to extract the x coordinates of the neurons to ensure that
highly-connected neurons remain close: the normalized second eigenvector of Q:

```python
eigvals, eigvecs = linalg.eig(Q)
x = Dinv2 @ eigvecs[:, 1]
```

Now it's just a matter of drawing the nodes and the links. We color them
according to the type stored in `neuron_types`:

```python
from matplotlib import pyplot as plt
from matplotlib import colors

def plot_connectome(neuron_x, neuron_y, links, labels, types):
    colormap = colors.ListedColormap([(1, 0, 0),
                                      (0, 0, 1),
                                      (0, 1, 0)])
    # plot neuron locations:
    plt.scatter(neuron_x, neuron_y, c=types, cmap=colormap, zorder=1)

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

**Exercise:** How do you modify the above code to show the affinity view in
Figure 2B from the paper?

### challenge: linear algebra with sparse matrices

The above code uses numpy arrays to hold the matrix and perform
the necessary computations. Because it is a small graph of fewer than 300
nodes, this is feasible. However, for larger graphs, it would fail.

In what follows, we will analyze the dependency graph for packages in the
Python Package Index, or PyPI, which contains over 75 thousand packages. To
hold the Laplacian matrix for this graph would take up
$8 \left(75 \times 10^3\right)^2 = 45 \times 10^9$ bytes, or 45GB,
of RAM. If you add to that the adjacency, symmetric adjacency, pseudoinverse,
and, say, two temporary matrices used during calculations, you climb up to
270GB, beyond the reach of most desktop computers.

However, we know that the dependency graph is *sparse*: packages usually depend
on just a few other packages, not on the whole of PyPI. So we can hold the
above matrices using the sparse data structures from `scipy.sparse` (see
Chapter 5), and use the linear algebra functions in `scipy.sparse.linalg` to
compute the values we need.

Try to explore the documentation in `scipy.sparse.linalg` to come up with a
sparse version of the above computation.

Hint: the pseudoinverse of a sparse matrix is, in general, not sparse, so you
can't use it here. Similarly, you can't get all the eigenvectors of a sparse
matrix, because they would together make up a dense matrix.

The solution is found in the following section, but we highly recommend that
you try it on your own!

## Community detection

Another application of the normalized graph Laplacian is in community
detection. Mark Newman published a
[seminal paper](http://www.pnas.org/content/103/23/8577.short)
on the topic in 2006. We'll apply it to the Python library dependency graph.

We've downloaded and preprocessed the data ahead of time, available as the file
`pypi-dependencies.txt` in the `data/` folder. The data consists of a list of
library-dependency pairs, one per line. The networkx library that we started
using in Chapter 3 makes it easy to build a graph from these data. We will use
a directed graph since the relationship is asymmetrical: one library depends
on the other, not vice-versa.

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
print('number of packages: ', dependencies.number_of_nodes())
print('number of dependencies: ', dependencies.number_of_edges())
```

What is the single most used Python package?

```python
print(max(dependencies.in_degree_iter(),
          key=lambda x: x[1]))
```

We're not going to cover it here, but `setuptools` is not a surprising winner
here. In fact, until we wrote this chapter, we had thought it was part of the
Python standard library, in the same category as `os`, `sys`, and others!

Since using setuptools is almost a requirement for being listed in PyPI, we are
actually going to remove it from the dataset, because it has too much influence
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

We can similarly find out the top-40 most depended-upon packages:

```python
packages_by_in = sorted(dependencies.in_degree_iter(),
                        key=lambda x: x[1], reverse=True)
print(packages_by_in[:40])
```

By this ranking, NumPy ranks 4 and SciPy 28 out of all of PyPI. Not bad!
Overall, though, one gets the impression that the web community dominates
PyPI. As we saw in the preface, this is expected, since the scientific Python
community is still growing!

However, the number of incoming links to a package doesn't tell the whole
story. As you might have heard, the key insight that drove Google's early
success was that important webpages are not just linked to by many webpages, but
also by *other* important webpages. This recursive definition implies that page
importance can be measured by the eigenvector corresponding to the largest
eigenvalue of the adjacency matrix.

We can apply this insight to the network of Python dependencies. First, we
ignore all the packages that are isolated:

```python
connected_packages = max(nx.connected_components(dependencies.to_undirected()),
                         key=len)
conn_dependencies = nx.subgraph(dependencies, connected_packages)
```

Next, we get the sparse matrix corresponding to the graph. Because a matrix
only holds numerical information, we need to maintain the list of package names
corresponding to the matrix rows/columns separately:

```python
package_names = conn_dependencies.nodes()
adjacency_matrix = nx.to_scipy_sparse_matrix(conn_dependencies)
```

Normally, the pagerank score would simply be the first eigenvector of the
adjacency matrix. However, this would give a huge advantage to "terminal" pages
that don't link anywhere else: if you think of the eigenvector as the result of
repeatedly hopping from one page to a random page it links to (this is the
effect of matrix multiplication), then "terminal" pages start to accumulate all
of the traffic they receive, without feeding into any other pages.

To counteract this effect, the pagerank algorithm uses a so-called "damping
factor", usually taken to be 0.85. This means that 85% of the time, the
algorithm follows a link at random, but for the other 15%, it randomly jumps to
another page.

This formulation implies the following mathematical relationship. If $A$ is the
adjacency matrix, let $K$ be the matrix with the degree of each page on the
diagonal, and 0 elsewhere. Then the transition matrix, for which entry $(i, j)$
contains the probability of going from page $i$ to page $j$, is $K^{-1}A$.
Then, with $M = (K^{-1}A)^T$, $\boldsymbol{r}$ being the vector of
pageranks, and $d$ being the damping factor, we have:

$$
\boldsymbol{r} = dM\boldsymbol{r} + \frac{1-d}{N} \boldsymbol{1}
$$

and

$$
(\boldsymbol{I} - dM)\boldsymbol{r} = \frac{1-d}{N} \boldsymbol{1}
$$

We can solve this equation using `scipy.sparse`'s conjugate gradient solver:

```python
from scipy import sparse
damping = 0.85
n = len(package_names)
degrees = np.asarray(adjacency_matrix.sum(axis=1)).ravel()
non_dangling = (degrees != 0)
degrees[non_dangling] = 1 / degrees[non_dangling]  # avoid divide-by-zero
degrees_matrix = sparse.diags([degrees], [0], adjacency_matrix.shape,
                              format='csr')
transition_matrix = (degrees_matrix @ adjacency_matrix).T

I = sparse.eye(n, format='csc')
```

With all the matrices we need in place, we can now solve the above equation:

```python
from scipy.sparse.linalg.isolve import bicg  # biconjugate gradient solver

pagerank, error = bicg(I - damping * transition_matrix,
                       np.full(n, (1-damping) / n),
                       maxiter=int(1e4))
print('error code: ', error)
```

As can be seen in the documentation for the `bicg` solver, an error code of 0
indicates that a solution was found. We now have the "dependency pagerank" of
packages in PyPI! Let's look at the top 40 packages:

```python
top = np.argsort(pagerank)[::-1]

print([package_names[i] for i in top[:40]])
```

NumPy actually falls one spot in this ranking, while SciPy drops out of the top
40 entirely! (
A graph of 90,000 nodes is a bit unwieldy to display, so we are actually going
to focus on the top 300, approximately matching the number of neurons in the
nematode brain.

```python
ntop = 300
top_package_names = [package_names[i] for i in top[:ntop]]
top_adj = adjacency_matrix[top[:ntop], :][:, top[:ntop]]
```


Where to get the data:
  * https://github.com/ogirardot/meta-deps/blob/master/PyPi%20Metadata.ipynb
  * https://ogirardot.wordpress.com/2013/01/05/state-of-the-pythonpypi-dependency-graph/
  * PyPI itself
  * github
  * something else?
  * `install_requires` vs DEPENDS.txt vs requirements.txt vs imports
  * http://furius.ca/snakefood/  (uses installed Python code. =\)

## Exercise

[ideas here?]

## Pagerank on above graph



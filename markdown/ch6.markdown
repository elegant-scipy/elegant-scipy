# Linear algebra in SciPy

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

[...]

## Community detection

Another application of the normalized graph Laplacian is in community
detection. Mark Newman published a
[seminal paper](http://www.pnas.org/content/103/23/8577.short)
on the topic in 2006. We'll apply it to the Python library dependency graph.

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



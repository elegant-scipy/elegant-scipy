Planning
=======
Just some notes to help us plan

Building chapters
=================
Using `notedown` to convert a markdown file to an IPython notebook, run it,
and then convert to html:

```console
$ notedown --match fenced --run markdown/chapter.markdown > ch2.ipynb
$ ipython nbconvert --to html ch2.ipynb
```

You can also run `make` to create all the chapters:

```console
$ make all
```

Deadlines
========
~~Mar: 2 chs: 3/15/15~~  
Jun: half chs: 6/1/15  
Oct: Ch5 (sparse): 10/15/15  
Nov: Ch6 (linalg): 11/15/15  
Dec: Ch7 (optimize): 12/15/15  
Dec: first full draft! 12/15/15  
Jan: All in:  1/15/16

Availabilities
-------------
Stéfan: ???

Harriet: In the US late Oct, all Nov, early Dec. Some writing possible in
second half of this.

Juan: Available for the rest of it!

Chapters/Ideas
==============
Based on: https://github.com/jni/my-documents/blob/oreilly-book-proposal/oreilly.markdown

*Preface/Introduction:* 

Why SciPy?

Who is this book for?

Ecosystem and community.

Conventions (e.g. NumPy docstring
conventions). Not sure if this fits here?

Definitions: What are SciPy NumPy matplotlib and pandas (or save some of these for when the are introduced?)

Getting Started:

Installation - Anaconda
Using IPythonNotebook and accessing the book materials

**Chapter 1-2:** Statistics, including numpy basics: memory layout, strides, F-
and C-contiguity, 1D, 2D arrays. Rough split between chapters 1 and 2:
broadcasting, advanced indexing.

**Chapter 3:** Image segmentation using `ndimage.generic_filter`. This includes
an introduction to the NumPy array, memory layout, strides, F- and
C-contiguity, 1D, 2D, and nD filters, and finally filtering to produce a graph
and segmentation from that graph. (This may be broken down into several
smaller chapters.)

**Chapter 4:** Fourier Transforms in 1D, 2D, and 3D. Scanning of a rock face.

**Chapter 5:** Use of sparse matrices to evaluate segmentation accuracy.

**Chapter 6:** Eigendecomposition. Ideas: Mark Newman's paper on
community-finding, pagerank for paper citations... Others?

**Chapter 7:** Optimization. Ideas: super-resolution microscopy.

**Chapter 8:** Streaming data. Piping a genome into a markov model.

**Chapter -1:** Contributing to the SciPy ecosystem: how to use git and github,
and follow best practices, to contribute to SciPy and related packages.

... Perhaps additional chapters with awesome examples using scikit-learn,
scikit-image, pandas, astropy, etc. The exact contents will be determined by
code submissions, which will be triaged to canvas as wide a footprint of SciPy
as possible.


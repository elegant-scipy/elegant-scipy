# Planning

Just some notes to help us plan.

## Installing dependencies

First, we build an isolated environment as not to interrupt any
existing setup you may have.  You can do this either via *conda* or *pip*.

### Conda

If you have a [conda](http://conda.pydata.org/miniconda.html) or
Anaconda installation, build a minimal environment to run everything
in the book as follows:

```console
conda env create --name elegant_scipy -f path/to/environment.yml
```

### Pip

If you do *not* have Conda installed, install **Python 3.5** and create a virtual
environment as follows:

```console
pyvenv create ~/envs/elegant_scipy  # Build the new environment -- do this only once
source ~/envs/elegant_scipy/bin/activate  # This switches into the new environment
```

Then, install the necessary requirements:

```
pip install -r requirements.txt
```

### Windows users

If you're using Windows, this is going to be a bit harder to get
going.  You need to make sure you have at least:

- [GitBash](https://git-scm.com/downloads) for unix utilities such as "make"
- [wget](https://sourceforge.net/projects/gnuwin32/files/wget/)

Even with these utilities installed, the build process is likely to
fail in unpredictable ways.  We suggest running virtual machine or
docker instance to build the book.

## Building chapters

We are using `notedown` to convert a markdown file to an IPython
notebook, run it, and then convert to html. For ease of use, this is
done using a Makefile.

You can use `make` to build all the chapters:

```console
$ make all
```

Or to build just an individual chapter, specify the file you wish to create:

```console
$ make html/ch1.html
```

To generate a zip file containing html of all chapters along with a table of contents (for easy sharing):

```console
$ make zip
```

## Deadlines

~~Mar: 2 chs: 3/15/15~~  
Jun: half chs: 6/1/15  
Oct: Ch5 (sparse): 10/15/15  
Nov: Ch6 (linalg): 11/15/15  
Dec: Ch7 (optimize): 12/15/15  
Dec: first full draft! 12/15/15  
Jan: All in:  1/15/16

## Availabilities

Stéfan: ???

Harriet: In the US late Oct, all Nov, early Dec. Some writing possible in
second half of this.

Juan: Available for the rest of it!

## Chapters/Ideas

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


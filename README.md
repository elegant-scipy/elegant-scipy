# Elegant SciPy

This is the online repository for the book
[Elegant SciPy](http://shop.oreilly.com/product/0636920038481.do),
written by Juan Nunez-Iglesias (@jni), Harriet Dashnow (@hdashnow), and Stéfan
van der Walt (@stefanv), and published by O'Reilly Media:

<a href="http://shop.oreilly.com/product/0636920038481.do">
<img src="https://github.com/elegant-scipy/elegant-scipy/blob/master/_images/cover.jpg?raw=true"
 alt="Elegant SciPy Cover" height=256>
</a>

## Using this book.

The text of the book (inside the `markdown` folder) is available under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
Public License (see `LICENSE.md`).

**All code** is available under the BSD 3-clause license (see
`LICENSE-CODE.md`). This includes:

- Python code embedded in the text of the book.
- Python code inside `scripts/`.
- Our `Makefile` and `.yml` files.

The authors also encourage educators to use this book in their own classrooms
for noncommercial instructional uses (i.e. for slide presentations in a
university lecture), provided that there is proper attribution to the O’Reilly
edition in each instance.

If you are unsure whether your use falls outside fair use or the permissions
given above, contact us at permissions@oreilly.com.

# Building the IPython notebooks

This book was written in markdown, with `notedown` and `jupyter nbconvert` used
to build the book. To recreate the book contents, install the dependencies,
(see below), then run `make all` from this directory (assuming you are using
Mac OS X or Linux).

For interactive exploration you probably don't want to pre-run all the code,
but rather create the notebooks with pre-populated input cells. To do this,
run, for example:

```console
notedown --match python /markdown/ch5.markdown --output ch5.ipynb
```

to build a Jupyter notebook containing Chapter 5, which you can then step
through by starting a notebook session in this directory:

```console
jupyter notebook
```

## Installing dependencies

First, we build an isolated environment as not to interrupt any
existing setup you may have.  This can be done using, e.g., Conda:

1. Install [conda](http://conda.pydata.org/miniconda.html) or Anaconda

2. Build an isolated environment called "elegant-scipy" and install the
   necessary dependencies:

```console
conda env create --name elegant-scipy -f path/to/environment.yml
```

### Windows

To build the full book on Windows, you will at a minimum need the following
additional software:

- [GitBash](https://git-scm.com/downloads) for Unix utilities such as "make"
- [wget](https://sourceforge.net/projects/gnuwin32/files/wget/)

Building the book on Windows is likely to be challenging because we developed
the build process on Mac and Linux. However, you should still be able to create
conda environments (see "Installing dependencies", above), and run notedown,
to make the Jupyter notebooks (see "Building the IPython notebooks", above).

If you encounter any problems, please
[raise an issue!](https://github.com/elegant-scipy/elegant-scipy/issues/new)

## Building the complete book

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

## Building HTMLBook

To export to HTMLBook, install
[htmlbook.js](https://www.npmjs.com/package/htmlbook) and then run `make
htmlbook`.

## Building with Docker

0. Switch to the directory containing this file
1. [Install docker-machine](https://docs.docker.com/machine/install-machine/)
2. Run `docker-compose up`

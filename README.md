# Elegant SciPy

## Installing dependencies

First, we build an isolated environment as not to interrupt any
existing setup you may have.  This can be done using, e.g., Conda:

1. Install [conda](http://conda.pydata.org/miniconda.html) or Anaconda

2. Build an isolated environment called "elegant_scipy" and install the
   necessary dependencies:

```console
conda env create --name elegant_scipy -f path/to/environment.yml
```

### Windows users

If you're using Windows, this is going to be a bit harder to get
going.  You need to make sure you have at least:

- [GitBash](https://git-scm.com/downloads) for unix utilities such as "make"
- [wget](https://sourceforge.net/projects/gnuwin32/files/wget/)

Even with these utilities installed, the build process is likely to
fail in unpredictable ways.  We suggest running virtual machine or
docker instance to build the book (see "Building with Docker" below).

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

## Building with Docker

0. Switch to the directory containing this file
1. [Install docker-machine](https://docs.docker.com/machine/install-machine/)
2. Run `docker-compose up`

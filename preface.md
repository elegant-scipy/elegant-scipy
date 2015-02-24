#Preface

> "Unlike the stereotypical wedding dress, it was—to use a technical term—elegant, like a computer algorithm that achieves an impressive outcome with just a few lines of code."

> -- Graeme Simsion,‎ *The Rosie Effect*

## Welcome to Elegant SciPy

NumPy and SciPy form the core of the Scientific Python ecosystem. These libraries already have excellent online documentation, so a complete reference would be pointless. Instead, in Elegant SciPy we present the best code built using these libraries, using it as motivation to teach readers who have never used them before.

The code examples will be chosen to highlight clever, elegant uses of advanced features of NumPy, SciPy, and related libraries. The beginning reader will learn to apply these libraries to real world problems using beautiful code. The book will start from first principles and provide all the necessary background to understand each example, including idioms, libraries (e.g. iterators), and scientific concepts. Examples will use actual scientific data.

This book will introduce readers to Scientific Python and its community, show them the fundamental parts of SciPy and related libraries, and give them a taste for beautiful, easy-to-read code, which they can carry with them to their practice.

## Who is this book for?
Elegant SciPy is intended to inspire you to take your Python to the next level. 
You will learn SciPy by example, from the very best code.

We have pitched this book towards people who have a decent beginner grounding in Python, and are now looking to do some more serious programming for their research.

We expect that you will be familiar with the Python programming language.
You have seen Python, and know about variables, functions, loops, and maybe even a bit of NumPy. But you don't know about the "SciPy stack" and you aren't sure about best practices. You might not have considered joining the SciPy community and contributing code.

Perhaps you are a scientist who has read some Python tutorials online, and have downloaded some analysis scripts from another lab or a previous member of their own lab, and have fiddled with them. But they don't have any solid concepts about what constitutes "good code".

If you are not yet familiar with Python basics, you might like to work through a beginner tutorial before tackling this book.
There are some great resources to get to get you started with Python, such as Software Carpentry (http://software-carpentry.org/).

Elegant SciPy is not a reference volume. Given the rapid rate of development, we would be remiss to write a reference book. Along the way, we will teach you how to use the internet as your reference.
Instead, in Elegant SciPy we will introduce you to the core concepts and capabilities of SciPy and related libraries. This is a book that you will read once, but may return to for inspiration (and maybe to admire some elegant code snippets!).

## Why SciPy?

NumPy and SciPy form the core of the Scientific Python ecosystem. The SciPy software library implements a set of functions for processing scientific data, such as statistics, signal processing, image processing, and function optimization. SciPy is built on top of the Python numerical array computation library NumPy. Building on NumPy and SciPy, an entire ecosystem of apps and libraries has grown dramatically over the past few years [plot?], spanning disciplines as broad as astronomy, biology, meteorology and climate science, and materials science.

This growth shows no sign of abating. For example, the Software Carpentry organization (http://software-carpentry.org/), which teaches Python to scientists, currently cannot keep up with demand, and is running "teacher training" every quarter, with a long waitlist.

In short, SciPy and related libraries will be driving much of scientific data analysis for years to come.

### Definitions: What is SciPy?

> "SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering."
> 
> -- http://www.scipy.org/

SciPy is a collection of Python packages. In Elegant SciPy we will see many of the main players in the SciPy ecosystem, such as:

* **NumPy**
provides efficient support for numerical computation, including linear algebra, random numbers, and Fourier transforms. 
A key feature of NumPy is that it allows you to define "N-dimensional arrays". These are data structures that can have any number of dimensions (more about this later). 
http://www.numpy.org/
* **SciPy library**
is a collection of efficient, user-friendly numerical algorithms. 
It also contains toolboxes for specific domains such as signal processing, integration, optimization, and statistics.
http://www.scipy.org/scipylib/index.html
* **Matplotlib**
is a powerful package for plotting in two dimensions (and basic 3D). It draws its name from the syntax that it shares with Matlab.
http://matplotlib.org/
* **IPython**
is an interactive interface for Python, so you can quickly interact with your data and test ideas. 
In particular, the IPython notebook runs in your browser and allows you to write code, text and mathematical expressions with inline plotting. These notebooks can easily be published with code alongside documentation and output, promoting reproducible research.
IPython supports multiple languages, allowing you to mix Python code with for example Cython, R, Octave, Bash, Perl and Ruby, in the same notebook.
http://ipython.org/
* **pandas**
provides fast, easy-to-use data structures, particularly to work with labelled data sets such as tables or relational databases, and manage time series.
It also has some handy data analysis tools such as for data parsing and cleaning, sliding windows, aggregation and plotting.
http://pandas.pydata.org/

## Ecosystem and community.
* Open Source
* GitHub
* Include nice touches such as good package names, such as airspeed velocity and sux.

Some of these here, or in a later chapter?:

* Contributing to the SciPy ecosystem: how to use git and github, and follow best practices, to contribute to SciPy and related packages.?

## Conventions 
### NumPy docstring conventions

## Getting Started
### Installation - Anaconda
### Accessing the book materials e.g. code

## Other stuff we could cover
related packages, where to get help

One more thing: Python 3 vs Python 2. We're going to use Python 3.4 to teach, but will by-and-large use code that is compatible with Python 2, assuming the user has the following preface (which might grow):

from __future__ import division, print_function
from six.moves import zip, map, filter
Pointing users to Python 2-3 resources online might be good. There's python-future.org, and Nick Coghlan's book-length guide on the topic. (Which is overkill.) Among others.
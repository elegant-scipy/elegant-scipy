#Preface

> "Unlike the stereotypical wedding dress, it was—to use a technical term—elegant, like a computer algorithm that achieves an impressive outcome with just a few lines of code."

> -- Graeme Simsion, *The Rosie Effect*

## Welcome to Elegant SciPy

This book is entitled Elegant SciPy. We’re going to spend rather a lot of time focusing on the “SciPy” bit of the title, so let’s take a moment to reflect on what we mean by elegant. There are plenty of manuals, tutorials and documentation websites out there describing the intricacies of SciPy. Elegant SciPy is not one of those. Instead of just teaching you how to write code that works, we want to step back and ask “is this code elegant?”.  

To explain what we mean by elegant code, we’ve drawn a rather apt quote from The Rosie Effect (hilarious book, by the way; go read The Rosie Project when you’re done with Elegant SciPy). Graeme Simsion twists the connotation of elegant around. Often, you would use elegant to describe the visual simplicity of something or someone stylish or graceful. Yet here, the protagonist expresses how pleasingly simple he finds the wedding dress by likening it to the experience of reading some delightfully concise code. That’s something we want you to get out of this book. To read or write a piece of elegant code, and feel calmed in the face of its beauty and grace (note, the authors may be prone to hyperbole). Perhaps you too will complement your significant other one how their outfit reminds you of that bit of code you saw in Elegant SciPy…

So, what makes code elegant? In scientific theory, you might use elegant to describe a pleasingly simple theorem. This is probably the definition of elegant that falls closest to Elegant SciPy. 

Elegant code is a pleasure to read, use and understand because it is:

* Simple
* Efficient
* Clear
* Creative
* Beautiful

By simple, I mean that elegant code achieves much in a few lines. Note that this is generally through abstraction and functions, *not* through just packing in a bunch of nested function calls! Elegant code is efficient, not only in terms of the number of keystrokes but also in terms of speed and memory. The packages and algorithms used take care to store and process data efficiently. Elegant code should be clear and easy to understand.
One way to ensure this is to comment often, and make sure your comments describe not just what your code is doing, but also how and why.Similarly, your variable and function names should be reasonably descriptive.To achieve simplicity, efficiency and clarity you will often need to use cleverness and creativity. 
For example, you might use a particularly efficient data structure in a new context to which it has not yet been applied.In many cases elegant code intrigues us, because it does something clever, approaching a problem in a new way, or just in a way that in retrospect is obvious in its simplicity.It is the culmination of the elements of elegant code just described that make your code "beautiful", a pleasure to write, read and to use. This is elegant code.

Now that we’ve dealt with the elegant part of the title, let’s bring back the SciPy.

SciPy forms the core of the Scientific Python ecosystem. 
These libraries already have excellent online documentation, so a complete reference would be pointless. Instead, in Elegant SciPy we present the best code built using these libraries, using it as motivation to teach readers who have never used them before.

The code examples will be chosen to highlight clever, elegant uses of advanced features of NumPy, SciPy, and related libraries. The beginning reader will learn to apply these libraries to real world problems using beautiful code. The book will start from first principles and provide all the necessary background to understand each example, including idioms, libraries (e.g. iterators), and scientific concepts. Examples will use actual scientific data.

This book will introduce readers to Scientific Python and its community, show them the fundamental parts of SciPy and related libraries, and give them a taste for beautiful, easy-to-read code, which they can carry with them to their practice.

When we started writing Elegant SciPy, we first put out a call to the community, asking Pythonistas to nominate the most elegant code they have seen. Like SciPy itself, we wanted Elegant SciPy to be driven by the community. The code herein is the best code the community has offered up for your pleasure.

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

## Getting Started

### Installation - Anaconda

Throughout this book we’re going to assume that you have Python 3.4 (or a later version) and have all the major SciPy packages installed: SciPy library, NumPy, Matplotlib, IPython and pandas.
The easiest way to get all of these components is to install the Anaconda Python distribution.
You can download Anaconda here: https://store.continuum.io/cshop/anaconda/. You will also find detailed installation instructions.

### Accessing the book materials e.g. code

All of the code from this book is available on our GitHub repository [link]

### The Great Cataclysm: Python 2 vs. Python 3

In your Python travels, you may have noticed that there is heated debate about which version of Python to use. You may wonder why they don't just use the latest.
In [year] the Python developers released a major update to Python, moving from Python 2 to 3. 
This was a breaking change, meaning that in many cases Python 2.x code can not be interpreted by Python 3.x. 
Python 2.7 was to be the last version of Python 2 to be released, and all development was continued on the Python 3 branch.

There were a number of reasons for making the leap to Python 3, one reason was a desire to standardise all the functions to use function notation, most notably: 

```
print "Hello World!"  # Python 2 print function
print("Hello World!") # Python 3 print function
```

Another breaking change was the way Python 3 treats integer division. In Python 2, an integer divided by another integer always returns an integer (and rounds down to do so). Python 3 performs integer division is a way that is more intuitive to new programmers, that is, an integer divided by an integer can return a float.

```
# Python 2
>>> 5/2
2
# Python 3
>>> 5/2
2.5
```

Another major advance of Python 3 over Python 2 was support for Unicode (an expanded character set).

```
# Unicode example
```

Although for the most part Pythonistas agree that the changes in Python 3 were sensible, and well-intentioned, the changes have inadvertently created a schism in the Python community.
The breaking changes in Python 3 mean that massive Python 2 code bases must be ported to Python 3 (which in many cases is not a trivial exercise). 
The inertia of these code bases is such that many developers have called for continued development of Python 2, incorporating the many non-breaking features that have been added to Python 3 in the time since it's inception. There has even be discussions of break-away development to continue on Python 2 by its proponents.

Given the divided state of the community, many developers now write code that is compatible with both Python 2 and 3. This can increase the potential impact of the code, but also creates an additional burden for the programmer.
New programmers must decide which version of Python to code in, which side of the fissure to join.

The debate rages, and it's up to you to choose a side. 
There are plenty of Python 2 vs. 3 resources online to help you choose your stance.
For example, you might like to check out python-future.org, and Nick Coghlan's book-length guide on the topic [ref].

In Elegant SciPy, we're going to use Python 3.4.
[why?]
However, if you're a Python 2 fan, by-and-large the code in this book is compatible with Python 2, assuming you have the following imports in your code:

```
from __future__ import division, print_function
from six.moves import zip, map, filter
```
As an example of the drama that has resulted from the Python 2 vs. 3 controversy, let us tell you about how this book was written. We decided on the following writing workflow:* Write the text and code of the book in Markdown format. Markdown is a simple - but sufficiently powerful - human readable text format that we can easily keep under Git version control (more on this later).* Convert the Markdown to IPython Notebook* Run the Python 3 code and save any outputs (like plots).We were pretty pleased with this plan, until we discovered that the software that converts Markdown to IPython is only able to produce Markdown format version 3. And Markdown v3 is only compatible with Python 2.7! This is what happens when half the community is using Python 2.7, and the other half is somewhere around Python 3.4...So we developed a new plan:* Write the text and code of the book in Markdown format.* Convert the Markdown to IPython Notebook v3.* Convert the IPython Notebook v3 to IPython Notebook v4 (which **is** compatible with Python 3).* Run the Python 3 code and save any outputs.And the moral of the story? A divided community is very inefficient!

## SciPy Ecosystem and Community

### The SciPy Ecosystem [some pun on ecosystem?]

[put what is SciPy here?]

### It Pays to be Open

The SciPy community embraces open source. The source code for all SciPy libraries is freely available to read, edit and reuse by anyone. 

So, why open? If you want others to use your code, one of the best ways to acheive this is to make it free. 
But what if you find some free software, but it doesn't do exactly what you want to acheive? 
If it's closed source, your only option is to use something else, email the developer and ask them to add a new feature (this often doesn't work!), or write it your self. 
If code is open source, it is much easier to add functionality to an existing code base than to write the whole thing again from scratch.

Similarly, if you find a bug in a piece of software, having access to the source code can make things a lot easier for both the user and the developer. 
With source code access, you can identify the source of the error, fix it, and then you have solved your problem. 
For close source code you would have to tell the developer about the bug, wait for them to get around to working on it, then probably have a lengthy email conversation back and forth trying to recreate the error.
From the developer's perspective, this is a big win. 
Imagine booting up your computer in the morning to find that one of your users has found a bug, reported it, tracked down the source of the error, fix it and then sent you the fix! This has certainly happened to me, and is quite a lovely experience.
Here's the last time this happened to me https://github.com/katholt/srst2/pull/32, but there are surely better examples.

Open for altruism

#### Open Source, Open Science

* Open source.
  * why open source is good: when things go wrong you can figure out why; when you need a new feature, you can easily hack it in. (if you have examples of this, that's even better. I do but will have to dig.)

#### Open Source Licenses

If you want others to be able to use your code, you *must* license it.
If you don't license your code, it is closed by default.
Even if you publish your code (for example by placing it in a publish GitHub repository), without a software license, no one is allowed to use, edit or redesribute your code.


Free and Open Source Software (FOSS) Licenses
 

Types of FOSS licenses:
* free
* copy left
 

  * licensing: get some inspiration from Jake's article: http://www.astrobetter.com/the-whys-and-hows-of-licensing-scientific-code/

* Licenses
  * SciPy community has adopted BSD which means lots of contributions from wide array of people, including many in industry, start-ups, etc.
  
All the code in this book that was written by us is available under a BSD license. Where we have sourced code snippets from other people, the code will generally be under an open license of some varity (although not necessarily BSD).


### GitHub: Taking Coding Social

* GitHub.
  * massive effect on open source contributions, specifically in Python. Quoting Jake again: http://jakevdp.github.io/blog/2012/09/20/why-python-is-the-last/ See his github plot there.
  * democratises development: anyone can come, fork a project, play to their heart's content, and eventually contribute via pull request.
  * introducing thousands to code review and continuous integration.

### A Touch of Whimsy with your Py

* Include nice touches such as good package names, such as airspeed velocity and sux.

** package names
  * airspeed velocity: continuous integration benchmarks, referencing Holy Grail http://spacetelescope.github.io/asv/using.html

  * sux: use Python 2 packages from Python 3. A play on "six" (the Python 2/3 compatibility package) and also great usage syntax: "sux.to_use('my_py2_package')" (This can also segue into The Great Cataclysm)

### Making your Mark on the SciPy Ecosystem

Some of these here, or in a later chapter?:

* Contributing to the SciPy ecosystem: how to use git and github, and follow best practices, to contribute to SciPy and related packages.?

** Contributing
[yes, leave this till the final chapter but mention it's coming, continuing from the mention of GitHub in "community", something like, "in the final chapter, we will show you how to contribute your new skills to the GitHub-hosted projects that comprise most of the scientific Python ecosystem."]


## Conventions 
### NumPy docstring conventions

## Other stuff we could cover
related packages, where to get help



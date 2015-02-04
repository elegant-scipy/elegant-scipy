#Preface


## Who is this book for?
Scientists seeking to take their Python to the next level will want to learn by example from the very best code.

I would think our main audience is people (mostly biologists) who have a decent beginner grounding in Python and are now having to do some more serious programming for their research.

A reader should be familiar with the Python programming language. There should be no other requirements to understand the book. We have two key audiences, who have similar levels of experience and understanding:

1. Those fresh out of a Software Carpentry Python tutorial. They signed up because they want to do more sophisticated things with their data than software such as Excel allows. They have seen Python, and have ideas about variables, functions, loops, and even a bit of NumPy, but they don't know about the SciPy stack and they certainly don't know best practices, how easy it is to join the community and contribute, etc.
2. The self-taught scientists. (This was me once upon a time.) They have read some Python tutorials online, and have downloaded some analysis scripts from another lab or a previous member of their own lab, and have fiddled with them. But they don't have any solid concepts about what constitutes "good code".

**Please provide some scenarios that indicate how the audience will use your book. For example, will readers refer to it daily as a reference? Will they read it once to learn the concepts and then refer to it occasionally?**

This will not be a reference volume. As I mentioned, the internet now provides a far better reference than any book could, especially considering the rapid pace of development in this field. Instead, a reader will learn the core concepts and capabilities of SciPy and related libraries by reading once, but may well return for inspiration to the final code examples themselves.


## Why SciPy?
**Summarize what the book is about, like you would pitch it to a potential reader on the back cover. What makes your book unique in the marketplace?**

NumPy and SciPy form the core of the Scientific Python ecosystem. They also have excellent online documentation, so a complete reference would be pointless. Instead, we present the best code using these libraries, using it as motivation to teach readers who have never used them before.

The examples will be chosen to highlight clever, elegant uses of advanced features of NumPy, SciPy, and related libraries. The beginning reader will learn not the functionality of the library, but its application to real world problems using beautiful code. The book will start from first principles and provide all the necessary background to understand each example, including idioms, libraries (e.g. iterators), and scientific concepts. Examples will use actual scientific data.

This book will introduce readers to Scientific Python and its community, show them the fundamental parts of SciPy and related libraries, and give them a taste for beautiful, easy-to-read code, which they can carry with them to their practice.

**Briefly explain the technology and why it is important.**

The SciPy software library implements a rather disjointed set of scientific data processing functions, such as statistics, signal processing, image processing, and function optimization. It is built on top of the numerical array computation library NumPy. On top of these two libraries, an entire ecosystem of apps and libraries has grown dramatically over the past few years (see the "how many people will use this technology" section), spanning disciplines as broad as astronomy, biology, meteorology and climate science, and materials science.

This growth shows no sign of abating. The Software Carpentry organization, which teaches Python to scientists, currently cannot keep up with demand, and is running "teacher training" every quarter, with a long waitlist. (I am enrolled in the current session, finishing this year.)

In short, SciPy and related libraries will be driving much of scientific data analysis for years to come.
### Definitions: What are SciPy and NumPy?

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
#Preface

> "Unlike the stereotypical wedding dress, it was—to use a technical term—elegant, like a computer algorithm that achieves an impressive outcome with just a few lines of code."

> -- Graeme Simsion, *The Rosie Effect*

Welcome to Elegant SciPy.
We’re going to spend rather a lot of time focusing on the “SciPy” bit of the title, so let’s take a moment to reflect on the "Elegant" bit.
There are plenty of manuals, tutorials and documentation websites out there that describe the SciPy library.
Elegant SciPy is not one of those.
Instead of just teaching you how to write code that works, we will inspire you to write code that rocks!

In The Rosie Effect (hilarious book; go read its prequel [The Rosie Project](https://en.wikipedia.org/wiki/The_Rosie_Project) when you’re done with Elegant SciPy), Graeme Simsion twists the conventions of the word "elegant" around.
Most would use it to describe the visual simplicity of something or someone stylish or graceful.
Say, the first iPhone.
Instead, just such an object (the wedding dress) is compared to delightfully concise code.
The kind of code that you rarely encounter, yet know immediately when you do.
We hope that you will know exactly what he means after reading this book.
That you will read or write a piece of elegant code, and feel calmed in the face of its beauty and grace.
(Note: the authors may be prone to hyperbole.)
Perhaps you too will compliment your significant other on how their outfit reminds you of that bit of code you saw in Elegant SciPy...

So, what makes code elegant?
Elegant code is a pleasure to read, use and understand because it is:

* Simple
* Efficient
* Clear
* Creative

By simple, I mean that elegant code achieves much in a few lines, through abstraction and functions, *not* through just packing in a bunch of nested function calls!
Elegant code is often efficient, not only in number of keystrokes but also in time and memory.
Elegant code should be clear and easy to understand.
It may take a minute to grok, but it should provide a crisp moment of understanding.
This can be done through clear variable and function names, and carefully crafted comments, explaining not what is being done (the code does this), but *why* it is being done.

To achieve simplicity, efficiency and clarity you will often need to use cleverness and creativity.
For example, you might use a particularly efficient data structure in a new context to which it has not yet been applied.
In many cases elegant code intrigues us, because it does something clever, approaching a problem in a new way, or just in a way that in retrospect is obvious in its simplicity.

It is the culmination of these elements of elegant code that make your code "beautiful", a pleasure to write, read and to use.
This is elegant code.

Now that we’ve dealt with the elegant part of the title, let’s bring back the SciPy.

SciPy is a library, an ecosystem, and a community.
Part of what makes these great is that they have excellent online documentation, rendering a complete reference book pointless.
Instead, in Elegant SciPy we want to present the best code built using these libraries.

The code we have chosen highlights clever, elegant uses of advanced features of NumPy, SciPy, and related libraries.
The beginning reader will learn to apply these libraries to real world problems using beautiful code.
And we use real scientific data to motivate our examples.

When we started writing Elegant SciPy, we put out a call to the community, asking Pythonistas to nominate the most elegant code they have seen.
Like SciPy itself, we wanted Elegant SciPy to be driven by the community.
The code herein is the best code the community has offered up for your pleasure.

## Who is this book for?

Elegant SciPy is intended to inspire you to take your Python to the next level.
You will learn SciPy by example, from the very best code.

We have pitched this book towards people who have a decent beginner grounding in Python, and are now looking to do some more serious programming for their research.
If you are not yet familiar with Python basics, work through a beginner tutorial before tackling this book.
There are some great resources to get to get you started with Python, such as Software Carpentry (http://software-carpentry.org/).

We expect that you will be familiar with the Python programming language.
You have seen Python, and know about variables, functions, loops, and maybe even a bit of NumPy.
But you don't know whether the "SciPy stack" is a library or a menu item from IHOP, and you aren't sure about best practices.
Perhaps you are a scientist who has read some Python tutorials online, and have downloaded some analysis scripts from another lab or a previous member of their own lab, and have fiddled with them.
And you might think that you are more or less alone when you learn to code SciPy.
You are not.

As we progress, we will teach you how to use the internet as your reference.
And we will point you to the mailing lists, repositories, and conferences where you will meet like-minded scientists who are a little farther in their journey than you.

This is a book that you will read once, but may return to for inspiration (and maybe to admire some elegant code snippets!).

## Why SciPy?

The NumPy and SciPy libraries form the core of the Scientific Python ecosystem.
The SciPy software library implements a set of functions for processing scientific data, such as statistics, signal processing, image processing, and function optimization.
SciPy is built on top of the Python numerical array computation library NumPy.
Building on NumPy and SciPy, an entire ecosystem of apps and libraries has grown dramatically over the past few years [plot?], spanning disciplines as broad as astronomy, biology, meteorology and climate science, and materials science.

This growth shows no sign of abating.
For example, the Software Carpentry organization (http://software-carpentry.org/), which teaches Python to scientists, currently cannot keep up with demand, and is running "teacher training" every quarter, with a long waitlist.

In short, SciPy and related libraries will be driving much of scientific data analysis for years to come.

### What is the SciPy Ecosystem?

> "SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering."
>
> -- http://www.scipy.org/

The SciPy ecosystem is a loosely defined collection of Python packages.
In Elegant SciPy we will see many of its main players:

* **NumPy** is the foundation of scientific computing in Python. It
provides efficient numeric arrays and  wide support for numerical computation, including linear algebra, random numbers, and Fourier transforms.
NumPy's killer feature are its "N-dimensional arrays", or `ndarray`.
These data structures store numeric values efficiently and define a grid in any number of dimensions (more about this later).
http://www.numpy.org/
* **SciPy**, the library,
is a collection of efficient, user-friendly numerical algorithms.
It contains toolboxes for specific domains such as signal processing, integration, optimization, and statistics.
http://www.scipy.org/scipylib/index.html
* **Matplotlib**
is a powerful package for plotting in two dimensions (and basic 3D). It draws its name from the syntax that it shares with Matlab.
http://matplotlib.org/
* **IPython**
is an interactive interface for Python, so you can quickly interact with your data and test ideas.
* **Jupyter**
The Jupyter notebook runs in your browser and allows you to write code in line with text and mathematical expressions, displaying the results of computation within the text.
This entire book has been written with Jupyter.
Jupyter started out as an IPython extension, but now supports multiple languages, including Cython, Julia, R, Octave, Bash, Perl and Ruby.
http://ipython.org/
* **pandas**
provides fast, easy-to-use data structures, particularly to work with labelled data sets such as tables or relational databases, and manage time series.
It also has some handy data analysis tools such as for data parsing and cleaning, sliding windows, aggregation and plotting.
http://pandas.pydata.org/
* **scikit-image**
provides image analysis tools that integrate cleanly with the rest of SciPy.

There are others, and we will see some of them too.
The takeaway here is that, although we will focus on NumPy and SciPy, it is the many surrounding packages that make Python a powerhouse for scientific computing.

## Getting Started

### Installation - Anaconda

Throughout this book we’re going to assume that you have Python 3.4 (or a later version) and have all the major SciPy packages installed: NumPy, SciPy, Matplotlib, IPython, scikit-image, and others.
We will list all of the requirements and the versions we used at the start of each chapter.
The easiest way to get all of these components is to install the Anaconda Python distribution.
You can download Anaconda here: http://continuum.io/downloads. You will also find detailed installation instructions.

If you find that you need to keep track of multiple versions of Python or different sets of packages, try Conda
(if you've just downloaded the Anaconda Python distribution then you already have it!).


### The Great Cataclysm: Python 2 vs. Python 3

In your Python travels, you may have already heard a few rumblings about which version of Python is better.
You may wonder why it's not just the latest version.

At the end of 2008, the Python core developers released a major update to Python, Python 3.0, a backwards-incompatible release.
In many (most?) cases, Python 2.6 or 2.7 code cannot be interpreted by Python 3.x without at least minor modification.
Python 2.7 was the last version of Python 2 to be released, and all development continues on the Python 3 branch (except for critical security updates).
To quote a certain genius, "this has made a lot of people very angry and been widely regarded as a bad move."

There will always be a tension between the march of progress and backwards compatibility.
There's as many opinions on the best way to move forward as there are developers in the world.
In this case, the Python core team decided that a clean break was needed to eliminate some inconsistencies in Python, and move it forward into the twenty-first century.
(Python 1.0 appeared in 1994, more than 20 years ago; a lifetime in the tech world.)

Here's one way in which Python has improved in turning 3:

```
print "Hello World!"   # Python 2 print statement
print("Hello World!")  # Python 3 print function
```

So what, right?
Why cause such a fuss just to add some parentheses!
Well, true, but what if you want to instead print to a different *stream*, such as *standard error*, the usual place for debugging information?

```
print >>sys.stderr, "fatal error"  # Python 2
print("fatal error", file=sys.stderr)  # Python 3
```

Ah, that's looking a bit more worthwhile.
What the hell is going on in that Python 2 statement?
The authors don't rightly know.

Another change is the way Python 3 treats integer division, which is the way most humans treat division.

```
# Python 2
>>> 5 / 2
2
# Python 3
>>> 5 / 2
2.5
```

Possibly the biggest improvement in Python 3 is its support for Unicode, a way of encoding text that allows one to use not just the English alphabet, but rather any alphabet in the world.
Python 2 allowed you to define a Unicode string, like so:

```python
beta = u"β"
```

But in Python 3, *everything* is unicode:

```python
β = 0.5
```

The Python core team decided, rightly, that it was worth supporting characters from all languages as first-class citizens in Python code.
This is especially true now, when most new coders are not coming from English-speaking countries.

Nevertheless, this update broke a lot existing code, and adding insult to injury, to date, Python 3 code runs slower than the equivalent Python 2 program — even when many already complained that Python was too slow.
That's certainly not motivating for someone who has to put work in to make their library work in Python 3, and it's left many continuing to use Python 2.7, refusing to upgrade.

Given the divided state of the community, many developers now write code that is compatible with both Python 2 and 3.

For new learners, the right thing to do is to use Python 3.
It is the future of the language, and there is no sense in adding to the dead weight of Python 2 code littering the interwebs (including much from your authors!).
In Elegant SciPy, we're going to use Python 3.4.

For more reading, you might like to check out Ed Schofield's python-future.org, and Nick Coghlan's book-length guide on the transition [^py3].

However, if you *must* use Python 2, you can make the code in this book compatible with Python 2, assuming you have the following imports at the start of your code:

```
from __future__ import division, print_function
from six.moves import zip, map, range, filter
```

## SciPy Ecosystem and Community

### The SciPy Ecosystem

SciPy is a major library with a lot of functionality.
Together with NumPy, it is one of Python's killer apps.
And such a killer app has launched a vast amount of related libraries that build on this functionality.
You'll encounter many of these throughout this book.

The creators of these libraries and many of their users gather at many events and conferences around the world.
These include the yearly SciPy conference in Austin, Texas, EuroSciPy, SciPy India, PyData and others.
We highly recommend attending one of these, and meeting the authors of the best scientific software in the Python world.
If you can't get there, or simply want a taste of these conferences, many [publish their talks online](https://www.youtube.com/user/EnthoughtMedia/playlists).

### Free and open-source software

The SciPy community embraces open source code.
The source code for nearly all SciPy libraries is freely available to read, edit and reuse by anyone.

So, why open?

If you want others to use your code, one of the best ways to achieve this is to make it free.
If you use closed source software, but it doesn't do exactly what you want to achieve, you're out of luck.
You can email the developer and ask them to add a new feature (this often doesn't work!), or write new software yourself.
If the code is open source, (using the skills you learn from this book) you can easily add or modify its functionality.

Similarly, if you find a bug in a piece of software, having access to the source code can make things a lot easier for both the user and the developer.
Even if you don't quite understand the code, you can usually get a lot further along in diagnosing the problem, and help the developer with fixing it.
It is usually a learning experience for everyone!

#### Open Source, Open Science

In scientific programming, all of these scenarios are extremely common and important: scientific software often builds on previous work, or modifies it in interesting ways.
And, because of the pace of scientific publishing and progress, much code is not thoroughly tested before release, resulting in minor or major bugs.

In science, another great reason for making code open source is to promote reproducible research.
Many of us have had the experience of reading a really cool paper, and then downloading the code to try it out on our own data.
Only we find, that the executable isn't compiled for our system. Or we can't work out how to run it.
Or it has bugs, missing features, or produces unexpected results.
By making scientific software open source, we not only increase the quality of that software, but we make it possible to see exactly how the science was done.
What assumptions were made, and even hard coded?
Open source helps to solve many of these issues.
It also enables other scientists to build on the code of their peers, fostering new collaborations and speeding up scientific progress.


#### Open Source Licenses

If you want others to be able to use your code, then you *must* license it.
If you don't license your code, it is closed by default.
Even if you publish your code (for example by placing it in a publish GitHub repository), without a software license, no one is allowed to use, edit or redistribute your code.

When choosing which of the many license options, you should first decide what you want people to be able to do with your code.
Do you want people to be able to sell your code for profit (or sell other code that uses your code), or do you want to restrict your code to be used only in free software?

There are two broad categories of FOSS license:

* Permissive
* Copy-left

A permissive license means that you are giving anyone the write to use, edit and redistribute your code in any way that they like.
This includes using your code as part of software that they are selling.
Some popular choices in this category include the MIT and BSD licenses.
The SciPy community has adopted the New BSD License (also called "Modified BSD" or "3-clause BSD").
Using such a license means receiving many code contributions from a wide array of people, including many in industry and start-ups.

Copy-left licenses also allow others use, edit and redistribute your code.
These licenses also prescribe that derived code must also be distributed under a copy-left license.
In this way, copy-left licenses restrict what users can do with the code.

The most popular copy-left license is the Gnu Public License, or GPL.
The main disadvantage to using a copy-left license is that you are effectively putting your code off-limits to any potential users or contributors from the private sector.
This can substantially reduce your user base and thus the success of your software.
In science, this probably means fewer citations.

For more help choosing a license, you might like to check out the Choose a License website http://choosealicense.com/.
For licensing in a scientific context, we recommend this blog post by Jake VanderPlas, Director of Research in the Physical Sciences at the University of Washington, and all around SciPy superstar:
http://www.astrobetter.com/the-whys-and-hows-of-licensing-scientific-code/.
In fact we quote Jake here, to drive home the key points of software licensing.

> ...if you only take three pieces of information away from the article, let them be these:
>
> 1. Always license your code.  Unlicensed code is closed code, so any open license is better than none (but see #2).
> 2. Always use a GPL-compatible license. GPL-compatible licenses ensure broad compatibility for your code, and include GPL, new BSD, MIT, and others (but see #3).
> 3. Always use a permissive, BSD-style license. A permissive license such as new BSD or MIT is preferable to a copyleft license such as GPL or LGPL.

> -- Jake VanderPlas http://www.astrobetter.com/the-whys-and-hows-of-licensing-scientific-code/

All the code in this book written by us is available under a BSD license.
Where we have sourced code snippets from other people, the code will generally be under an open license of some variety (although not necessarily BSD).

### GitHub: Taking Coding Social

We've talked a little about releasing your source code under an open source license.
This will hopefully result in huge numbers of people downloading your code, using it, fixing bugs and adding new features.
Where will you host your code so people can find it?
How will those bug fixes and features get back into your code? How will you keep track of all the issues and changes?
You can imagine how this could get out of control quite quickly.

Enter GitHub.

GitHub (https://github.com/) is a website for hosting, sharing and developing code.
It is based on Git version control software (http://git-scm.com/).
We will help you get started in GitHub land, but there are some great resources for a more in-depth experience, e.g
**[ED NOTE, reference GitHub book http://shop.oreilly.com/product/0636920033059.do]**

GitHub has had a massive effect on open source contributions, particularly in Python.
GitHub allows users to publish code publicly.
Anyone can come along and create a copy (fork) of the code and edit it to their heart's content.
They can eventually contribute those changes back into the original by creating a pull request.
There are some nice features like managing issues and change requests, as well as who can directly edit your code.
You can even keep track of editing statistics and other fun stats.
There are a whole bunch of other great GitHub features, but we will leave many them for you to discover and some for you to read in later chapters.
In essence, GitHub has democratized software development. It has substantially reduced the barrier to entry.

![The impact of GitHub](https://jakevdp.github.io/figures/author_count.png)
**[ED NOTE, this is a nice plot showing how Python development takes off after introduction of GitHub, but we need to email Jake VanderPlas to ask if we can use it. http://jakevdp.github.io/blog/2012/09/20/why-python-is-the-last/]**

### Making your Mark on the SciPy Ecosystem

Once you have become more comfortable with SciPy and started using it for your research, you may find that a particular package is lacking a feature you need, or you think that you can do something more efficiently, or perhaps you've found a bug.
When you reach this point, it's time to start contributing to the SciPy Ecosystem.

We strongly encourage you to try doing this.
The community lives because people are willing to share their code and improve existing code.
Beyond any altruistic reasons for contributing, there are some very practical personal benefits.
By engaging with the community you will become a better coder.
Any code you contribute will generally be reviewed by others and you will receive feedback.
As a side effect, you will learn how to use Git and GitHub, which are very useful tools for maintaining and sharing your own code.
You may even find interacting with the SciPy community provides you with a broader scientific network, and surprising career opportunities.

Later in Elegant SciPy we will show you how to contribute your new skills to the GitHub-hosted projects that comprise most of the scientific Python ecosystem.
In the meantime we want you to start thinking about being more than just a SciPy user.
You are joining a community and we hope you will keep making it a better place for all scientific coders.

### A Touch of Whimsy with your Py

In case you were worried that the SciPy community might be an imposing place to the newcomer, remember that it is made of people like you, scientists, usually with a great sense of humor.

In the land of Python, it is inevitable that you find some Monty Python references.
The package Airspeed Velocity (http://spacetelescope.github.io/asv/using.html) measures your software's speed (more on this later), and references "what is the airspeed velocity of an unladen swallow?" from *Monty Python and the Holy Grail*.

Another amusingly titled package is "Sux", which allows you to use Python 2 packages from Python 3.
This is a play on "six", which lets you use Python 3 syntax in Python 2, with a New Zealand accent.
Sux syntax makes it less frustrating to use Python 2-only packages after you've moved to Python 3:

```
import sux
sux.to_use('my_py2_package')
```

In general, Python library names can be a riot, and we hope you'll enjoy your time coming up with some!

## Getting Help

My first step when I get stuck is to Google the task that I am trying to achieve,
or the error message that I got.
This generally leads me to [Stack Overflow](http://stackoverflow.com/),
an excellent question and answer site for programming.
If you don't find what you're looking for immediately try generalizing your search terms to find someone who is have a a similar problem.
Sometimes, you realize that you might be the first person to have this specific question (this is particularly likely when you are using a brand new package), but not all is lost!
As mentioned above, the SciPy community is a friendly bunch, and can be found scattered around various parts of the interwebs.
Your next point of call is to Google "`<library name> mailing list`", and find
an email list to ask for help.
Library authors and power users read these regularly, and are very welcoming to newcomers.
Note that it is common etiquette to *subscribe* to the list before posting.
If you don't, it usually means someone will have to manually check that your email isn't spam before allowing it to be posted to the list.
It may seem annoying to join yet another mailing list, but we highly recommend it: it is a great place to learn a lot, bit by bit!

### Accessing the book materials e.g. code

All of the code from this book is available on our GitHub repository [link]().

## Diving in

We've brought together some of the most elegant code offered up by the SciPy community.
Along the way we are going to explore some real-world scientific problems that SciPy can solve.
This book is also a glimpse into a welcoming scientific coding community that wants you to join in.

Welcome to Elegant SciPy.

Now, let's go write some code!


[^py3]: http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html

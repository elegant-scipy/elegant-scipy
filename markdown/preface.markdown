#Preface

> "Unlike the stereotypical wedding dress, it was—to use a technical term—elegant, like a computer algorithm that achieves an impressive outcome with just a few lines of code."

> -- Graeme Simsion, *The Rosie Effect*

Welcome to *Elegant SciPy*.
We’re going to spend rather a lot of time focusing on the “SciPy” bit of the title, so let’s take a moment to reflect on the "Elegant" bit.
There are plenty of manuals, tutorials and documentation websites out there that describe the SciPy library.
Elegant SciPy goes further.
Beyond teaching you how to write code that works, we will inspire you to write code that rocks!

In The Rosie Effect (hilarious book; go read its prequel [The Rosie Project](https://en.wikipedia.org/wiki/The_Rosie_Project) when you’re done with Elegant SciPy), Graeme Simsion twists the conventions of the word "elegant" around.
Most would use it to describe the visual simplicity, with style or grace — say, the first iPhone.
Instead, *The Rosie Effect* uses a computer algorithm to *define* elegance.
We hope that you will understand after reading this book: experience a piece of elegant code and feel calmed in the glow of its beauty and grace.
(Note: the authors may be prone to hyperbole.)
Perhaps you, too, will compliment your significant other on how their outfit reminds you of that bit of code you saw in Elegant SciPy...

So, what makes code elegant?
Elegant code is a pleasure to read, use and understand because it is:

* Simple
* Efficient
* Clear
* Creative

Elegant code achieves much in a few lines, through abstraction and functions, *not* through just packing in a bunch of nested function calls!
Elegant code is often efficient, not only in number of keystrokes but also in time and memory.
Elegant code should be clear and easy to understand.
It may take only a minute to grok and yet provides a crisp moment of understanding.
This can be done through clear variable, function names, and carefully crafted comments that explain, not merely describe, the code.

Writing simple, efficient, clear code requires significant creativity.
In the *New York Times*, software engineer J. Bradford Hipps [argued](http://www.nytimes.com/2016/05/22/opinion/sunday/to-write-software-read-novels.html) that "to write better code, [one should] read Virginia Woolf."
You might use a particularly efficient data structure in a new, unexpected context. **THE ARTICLE ITSELF DOES NOT MENTION VIRGINIA WOOLF AT ALL; BETTER TO QUOTE DIRECTLY FROM THE ARTICLE, FOLLOWED BY A BRIEF ANALYSIS**

In many cases elegant code intrigues us because it does something clever, approaching a problem in a new way, or just in a way that in retrospect is obvious in its simplicity.
It is the culmination of these elements of elegant code that make your code "beautiful", a pleasure to write, read and to use.
This is elegant code. **AWKWARD AND UNNECESSARILY LONG PARAGRAPH. WHAT EXACTLY IS ELEGANT CODE?**

What, then, is "SciPy", and how could it be made elegant?

SciPy is a library, an ecosystem, and a community.  Part of what makes
SciPy great is that it has excellent online documentation and
tutorials (see, e.g., https://docs.scipy.org and
http://www.scipy-lectures.org/), rendering Just Another Reference book
pointless. Instead, *Elegant SciPy* presents the best code built on SciPy.

The codes we have chosen highlight clever, elegant uses of NumPy, SciPy, and related libraries.
Through *Elegant SciPy*, the novice reader will learn to apply these libraries to real world problems using beautiful codes. Beautiful codes that analyze real, scientific data.

When we started writing *Elegant SciPy*, we put out a call to the **WHICH?** community, asking Pythonistas to nominate the most elegant code they have seen.
Like SciPy itself, we wanted Elegant SciPy to be driven by the community.
The codes herein are the best codes the community has offered up for your pleasure.

## Who is this book for?

*Elegant SciPy* aims to inspire you to take your Python **ability? skills?** to the next level.
You will learn SciPy by example, from the very best codes.

We wrote the book for those who have some exposure to Python and are interested in serious programming for scientific research.
If you are not yet familiar with the basics of Python, work through a tutorial before tackling this book.
There are great resources to introduce you to Python, such as Software Carpentry (http://software-carpentry.org/).

We expect that you are familiar with the Python programming
language.  You know about variables, functions,
loops and, perhaps, a bit of NumPy.  Perhaps you've even honed your
Python skills with more advanced material, such as *Fluent Python*
(http://shop.oreilly.com/product/0636920032519.do).

But perhaps you don't know if the "SciPy stack" is a library or a menu item from International House of Pancakes.
Perhaps you aren't sure about best practices.
Perhaps you are a scientist in possession of an analysis script written by someone else, poking at it with the guidance of online Python tutorials. 
And, perhaps, you might think that you are alone as you attempt to learn SciPy.

You are not.

As we progress, we will teach you how to use the internet as your reference.
We will point you to the mailing lists, repositories, and conferences where you will meet like-minded scientists who are a little further in their journey than you.

Read the book, and return to it for inspiration. (Return to it again, perhaps, to admire some elegant code snippets?)

## Why SciPy?

The NumPy and SciPy libraries make up the core of the Scientific Python ecosystem.
The SciPy software library implements a set of functions for processing scientific data, such as statistics, signal processing, image processing, and function optimization.
SciPy is built on top of NumPy, the Python numerical array computation library.
Building on NumPy and SciPy, an entire ecosystem of apps and libraries has grown dramatically over the past few years, spanning disciplines as broad as astronomy, biology, meteorology and climate science, and materials science.

This growth shows no sign of abating.
For example, the Software Carpentry organization (http://software-carpentry.org/), which teaches Python to scientists, currently cannot keep up with demand and is running "teacher training" every quarter —  with a long waitlist.

SciPy and related libraries will be driving much of scientific data analysis for years to come.

### What is the SciPy Ecosystem?

> "SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering."
>
> -- http://www.scipy.org/

The SciPy ecosystem is a loosely defined collection of Python packages.
In *Elegant SciPy* we will meet its main players [^learn]:

* **NumPy** is the foundation of scientific computing in Python. It
provides efficient numeric arrays and wide support for numerical computation, including linear algebra, random numbers, and the Fourier transform.
NumPy's killer feature is its "N-dimensional arrays", or `ndarray`.
These data structures store numeric values efficiently and define a grid in any number of dimensions (more about this later).
http://www.numpy.org/
* **SciPy**, the library, is a collection of domain-specific efficient numerical algorithms **THERE IS NO SUCH THING AS A "USER-FRIENDLY" ALGORITHM; UX DEPENDS ON IMPLEMENTATIONS, NOT ALGORITHMS** for signal processing, integration, optimization, statistics, and so on.
http://www.scipy.org/scipylib/index.html
* **Matplotlib** is a powerful package for plotting in two dimensions and, to a limited extent, in three dimensions. It draws its name from its MATLAB-like syntax.
http://matplotlib.org/
* **IPython** is an interactive interface for Python, which can be used for testing ideas and interacting with data.
* **Jupyter**, the Jupyter notebook, runs in your browser and allows you to write code in line with text and mathematical expressions, displaying the results of computation within the text. (This entire book has been written with Jupyter.) Jupyter started out as an IPython extension but now supports multiple languages, including Cython, Julia, R, Octave, Bash, Perl and Ruby.
http://jupyter.org
* **pandas** provides fast data structures **AGAIN, THERE IS NO SUCH THING AS AN "EASY-TO-USE" DATA STRUCTURE**, particularly for labelled data sets — such as tables or relational databases — and for managing time series.
It also has handy data analysis tools for data parsing and cleaning, sliding windows, aggregation and plotting.
http://pandas.pydata.org/
* **scikit-learn** provides a unified interface to machine learning algorithms. **NO FURTHER EXPLANATION?**
* **scikit-image** provides image analysis tools that integrate cleanly with the rest of SciPy. **NO FURTHER EXPLANATION?**

Several other packages will be introduced throughout the book as well.



Despite the book's focus on NumPy and SciPy, it is the many surrounding packages that make Python a powerhouse for scientific computing.

[^learn]: You can learn more about many of the tools mentioned here at http://www.scipy-lectures.org/ or through field-specific guides such as *Effective Computation in Physics* (http://physics.codes/).


## Installing Python - Anaconda

Throughout this book, we will assume that you have Python 3.5 (or a later version) along with all the major SciPy packages. **THEY HAVE ALREADY BEEN INTRODUCED ABOVE. NO NEED TO REPEAT**
The list of required packages, along with version numbers, can be found in **envrionment.yml**, packaged with the data for this book.
The easiest way to get all of these components is to install the Anaconda Python distribution.
You can download Anaconda here: http://continuum.io/downloads.
There, you will also find detailed installation instructions.

If you find that you need to keep track of multiple versions of Python or different sets of packages, try Conda.
(If you've just downloaded the Anaconda Python distribution then you already have it!)
**environment.yml** can be passed to conda to install the right versions of everything in one go.

## The Great Cataclysm: Python 2 vs. Python 3

In your Python journey, you might have already heard rumblings about which version of Python is better.
Why not the latest version, you might have wondered.

At the end of 2008, the Python core developers released a major update to Python, Python 3.0, a backwards-incompatible release.
In most cases, Python 2.6/2.7 code must be modified to be interpreted by Python 3.x.
Python 2.7 is the last version of Python 2 to be released, and all development continues only on the Python 3 branch, except for critical security updates.
To quote a certain genius, "this has made a lot of people very angry and been widely regarded as a bad move."

There will always be a tension between the march of progress and backwards compatibility.
There are as many opinions on the best way to move forward as there are developers in the world.
In this case, the Python core team decided that a clean break was needed to eliminate some inconsistencies in Python, and move it forward into the Twenty-First Century.
(Python 1.0 appeared in 1994, more than 20 years ago; a lifetime in the tech world.)

Here's one way in which Python has improved in turning 3:

```
print "Hello World!"   # Python 2 print statement
print("Hello World!")  # Python 3 print function
```

So what, right?
Why cause such a fuss just to add some parentheses?
Well, true, but what if you want to instead print to a different *stream*, such as *standard error*, the usual place for debugging information?

```
print >>sys.stderr, "fatal error"  # Python 2
print("fatal error", file=sys.stderr)  # Python 3
```

Ah, that looks a bit more worthwhile.
What the hell is going on in that Python 2 statement?
Rightly, the authors don't know.

Another change is the way Python 3 treats integer division, which is the way most humans treat division.

```
# Python 2
>>> 5 / 2
2
# Python 3
>>> 5 / 2
2.5
```

We were also pretty excited about the new `@` *matrix multiplication* operator introduced in Python 3.5 in 2015.
Check out chapters 5 and 6 for some examples of this operator in use!

Possibly the biggest improvement in Python 3 is its support for Unicode, a way of encoding text that allows not just the English alphabet but any alphabet in the world.
Python 2 allows you to define a Unicode string, like so:

```python
beta = u"β"
```

But, in Python 3, *everything* is Unicode:

```python
β = 0.5
print(2 * β)
```

The Python core team decided, rightly, that it was worth supporting characters from all languages.
This is especially true now, when most new coders are coming from non-English-speaking countries.

Nevertheless, the introduction of Python 3 broke a lot existing code. Adding insult to injury, Python 3 codes run slower than the equivalent Python 2 ones to this date — and there were already many complaints about Python being too slow.
This is certainly not motivating for someone who has to put work in to make their library work in Python 3. Indeed,  many continue to use Python 2.7, refusing to upgrade.

Given the divided state of the community, many developers now write code that is compatible with both Python 2 and 3.

For new learners, the right thing to do is to use Python 3.
It is the future of the language, and there is no sense in adding to the dead weight of Python 2 code littering the interwebs (including much from your authors!).
In *Elegant SciPy*, we will use Python 3.5.

If you *must* use Python 2, adding the following makes most of the codes in the book compatible with Python 2:

```
from __future__ import division, print_function
from six.moves import zip, map, range, filter
```

A word of warning for Python 2.7 enthusiasts: chapters 5 and 6 will rely heavily on the new `@` *matrix multiplication* operator (mentioned above), which is only available in Python 3.5+.
Using `@` in an earlier version of Python will give you a nasty syntax error.
To make the codes in these chapters work with Python 2.7, you will have to use the `.dot` method of NumPy arrays and SciPy matrices, which is decidedly inelegant.
More on how to do this in chapter 5.

For more on this topic, you might want to check out Ed Schofield's python-future.org, and Nick Coghlan's book-length guide on the transition [^py3].

## SciPy Ecosystem and Community

SciPy is a major library with a lot of functionality, a true killer app when combined with NumPy.
It has launched a vast number of related libraries that build on this functionality, many of which you will encounter throughout this book.

The creators of these libraries, as well as many of the users, gather at events and conferences around the world.
These include the yearly SciPy conference in Austin, EuroSciPy, SciPy India, PyData, and others.
We highly recommend attending one of these and meeting the authors of the best scientific software in the Python world.
If you can't get to one of these or simply want a taste of these conferences, check out the [talks published online](https://www.youtube.com/user/EnthoughtMedia/playlists).

### Free and open-source software

The SciPy community embraces open-source software development..
The source codes for nearly all SciPy libraries are freely available to read, edit, and reuse by anyone.

So, why open?

If you want others to use your code, make it free. **FREE OR OPEN?**
If you use closed-source software and it doesn't do exactly what you want, then you're out of luck.
You could email the developer and ask them to add a new feature — this rarely works! — or write new software yourself.

If, however, the code is open source, you could easily add or modify its functionality using the skills you learn from this book. Having access to the source code can expedite the debugging process as well.
Even if you don't quite understand the code, you can usually get a lot further along in diagnosing the problem and help the developer fix it — a learning experience for everyone!

#### Open Source, Open Science

In scientific programming, all of these **WHICH?** scenarios are extremely common and important: scientific software often builds on previous work, or modifies it in interesting ways.
And, the pace of scientific research renders thorough, pre-release testing of codes impossible, resulting in many bugs.

Another great reason for making scientific codes open source is to promote reproducible research.
Many of us have had the experience of reading a cool paper and downloading the accompanying code to try it out, only to find out that the executable isn't compiled for our system. Perhaps the software isn't easy to run.
Perhaps it has bugs or missing features. Perhaps it produces unexpected results.

Making scientific software open-source not only improves the quality of that software but also shows us how the science was done.
What assumptions were made, and even hard-coded?
Open source helps to solve many of these issues. **WHICH ISSUES?**
It also enables other scientists to build on the code of their peers, fostering new collaborations and speeding up scientific progress.

#### Open Source Licenses

If you want others to be able to use your code, then you *must* license it.
If you don't license your code, it is closed by default.
Even if you publish your code (for example by placing it in a public GitHub repository), without a software license, no one is allowed to use, edit or redistribute your code.

When choosing a license, you must first decide what you want people to be able to do with your code.
Do you want people to be able to sell your code, or software that includes your code, for profit? Do you want to restrict your code to be used only in free software?

There are two broad categories of FOSS **YOU HAVE NOT DEFINED FOSS YET. SPELL IT OUT.** license:

* Permissive
* Copy-left

A permissive license means that you are giving everyone the right to use, edit, and redistribute your code, even as part of commercial software.
Using a permissive license likely results in receiving code contributions from a wide array of people, including industry and start-up developers. 
Popular choices in this category include the MIT and BSD licenses.
The SciPy community has adopted the 3-clause BSD License, also known as the New BSD License or the Modified BSD License.

Copy-left licenses also allow others use, edit and redistribute your code.
These licenses, however, prescribe that any derived code must also be distributed under a copy-left license, restricting what developers can do with the code.
The main disadvantage of using a copy-left license that it often results in barring the private sector from using or contributing to your software.
In science, this could mean fewer citations.
The most widely used copy-left license is the GPL, the GNU General Public License. Popular examples of GPL-licensed include **the Linux kernel? GCC? MySQL? Emacs?**

For additional help in choosing a license, check out http://choosealicense.com/.
For information on licensing in scientific contexts, we recommend this blog post by Jake VanderPlas, Director of Research in the Physical Sciences at the University of Washington and an all-around SciPy superstar:
http://www.astrobetter.com/the-whys-and-hows-of-licensing-scientific-code/.
In fact, we quote Jake below to drive home the key points of software licensing:

> ...if you only take three pieces of information away from the article, let them be these:
>
> 1. Always license your code.  Unlicensed code is closed code, so any open license is better than none (but see #2).
> 2. Always use a GPL-compatible license. GPL-compatible licenses ensure broad compatibility for your code, and include GPL, new BSD, MIT, and others (but see #3).
> 3. Always use a permissive, BSD-style license. A permissive license such as new BSD or MIT is preferable to a copyleft license such as GPL or LGPL.

> -- Jake VanderPlas http://www.astrobetter.com/the-whys-and-hows-of-licensing-scientific-code/

The codes in this book written by the authors are available under a BSD license.
Code snippets from contributors are generally under an open-source license of some variety, although not necessarily BSD.

For your own code, we recommend that you follow the practices of your
community. In Scientific Python, this means 3-clause BSD, while the R language,
for example, has adopted the GPL license. **THE LANGUAGE DOES NOT ADOPT A LICENSE. THE COMMUNITY, PERHAPS?**

### GitHub: Taking Coding Social

We've talked a little about releasing your source code under an open source license.
This will hopefully result in huge numbers of people downloading your code, using it, fixing bugs and adding new features.
Where will you put your code so people can find it?
How will those bug fixes and features get back into your code? How will you keep track of all the issues and changes?
You can imagine how this could get out of control quite quickly.

Enter GitHub.

GitHub (https://github.com/) is a website for hosting, sharing and developing code.
It is based on the Git version control software (http://git-scm.com/).
We will help you get started in the GitHub land, but there are some great resources for a more in-depth experience, e.g
**[ED NOTE, reference GitHub book http://shop.oreilly.com/product/0636920033059.do]**

GitHub has had a massive effect on open source contributions, particularly in Python. **IS THERE EVIDENCE THAT GITHUB HAS PARTICULARLY BEEN USEFUL FOR PYTHON?**
**TRANSITION SENTENCE NEEDED**
GitHub allows users to publish code.
Once a code is published, anyone can come along and create a copy (called a *fork*) of the code and edit it to their heart's content.
They can even contribute those changes back into the original by creating a *pull request*.
~~There are some nice features like managing issues and change requests, as well as who can directly edit your code.
You can even keep track of edits, contributors and other fun stats.~~
There are a whole bunch of other great GitHub features, but we will leave many them for you to discover and some for you to read in later chapters.

In essence, GitHub has democratized software development by substantially reducing the barrier to entry.

![The impact of GitHub](https://jakevdp.github.io/figures/author_count.png)
**[Used with permission of the author, Jake VanderPlas]**

### Make your Mark on the SciPy Ecosystem

As you gain more experience with SciPy, maybe you'll find that a particular package is lacking a feature you need. Maybe you found a bug. Maybe you think you can improve the package.
When you reach this point, it's time to start contributing to the SciPy Ecosystem.
The community lives on because of shared codes and continued improvements

Contributing to SciPy brings personal benefits as well.
By engaging with the community, you will become a better coder, as
your contributions will be reviewed with feedback.
Along the way, you will learn to use Git and GitHub, which are useful tools for maintaining and sharing your own code.
You may even find that interacting with the SciPy community provides you with a broader scientific network and surprising career opportunities.

Later in *Elegant SciPy*, we will show you how to use your new skills to contribute to the GitHub-hosted projects.
In the meantime, we want you to start thinking about being more than just a SciPy user.
You are joining a community, and we hope you will keep making it a better place for all scientific coders.

### A Touch of Whimsy with your Py

In case you were worried that the SciPy community might be an imposing place to the newcomer, remember that it is made of people like you, scientists, usually with a great sense of humor.

In the land of Python, it is inevitable that you find some Monty Python references.
The package Airspeed Velocity (http://spacetelescope.github.io/asv/using.html) measures your software's speed (more on this later). The name is a reference to the line "what is the airspeed velocity of an unladen swallow?" from *Monty Python and the Holy Grail*.

Another amusing package title is "Sux", which allows you to use Python 2 packages from Python 3.
This is a play on "six", which lets you use Python 3 syntax in Python 2, with a New Zealand accent.
Sux syntax makes it less frustrating to use Python 2-only packages after you've moved to Python 3:

```
import sux
p = sux.to_use('my_py2_package')
```

In general, Python library names can be a riot, and we hope you'll enjoy your time coming up with new ones!

## Getting Help

When we get stuck, we Google the task at hand or the error message on the screen.
This generally leads us to [Stack Overflow](http://stackoverflow.com/),
an excellent question-and-answer site for programmers.
If you don't find what you're looking for immediately, try broadening your search terms to find a relevant post.

Sometimes, you might actually be the first person to have this specific question —this is particularly likely when you are using a brand new package — but not all is lost!
As mentioned above, the SciPy community is a friendly bunch, and can be found scattered around various parts of the interwebs. **THIS IS CUTE BUT IS THIS SENTENCE NECESSARY?**

Your next point of call is to Google "`<library name> mailing list`" to find
a mailing list.
Library authors and power users read these mailing lists regularly and are very welcoming to newcomers.
Remember, it is common etiquette to *subscribe* to the list before posting.
If you don't, someone will often have to manually check that your email isn't spam before allowing it to be posted to the list.
Joining yet another mailing list might be annoying, but we highly recommend it. It is a great way to learn! 

### Accessing the book materials

All of the code from this book is available on our GitHub repository [link](). **LINK?**

## Diving in

We've brought together some of the most elegant codes offered up by the SciPy community.
Along the way, we will explore some real-world scientific problems that SciPy can solve.
This book is also a glimpse into a welcoming scientific coding community that wants you to join in.

Welcome to *Elegant SciPy*. Let's write some code!


[^py3]: http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html

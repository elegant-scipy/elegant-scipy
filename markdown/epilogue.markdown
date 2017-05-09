# Epilogue

> Quality means doing it right when no one is looking.
>
> — Henry Ford

Our main goal with this book was to promote elegant uses of the NumPy and SciPy
libraries. While teaching you how to do effective scientific analysis with SciPy,
we hope to have inspired in you the feeling that quality code is something
worth striving for.

## Where to next?

Now that you know enough SciPy to analyze whatever data gets thrown your way,
how do you move forward? We said when we started that we couldn't hope to cover
all there is to know about the library and all its offshoots. Before we part ways,
we want to point you to the many resources available to help you.

### Mailing lists

We mentioned in the preface that SciPy is a community. A great way to continue
learning is to subscribe to the main mailing lists for NumPy, SciPy, pandas,
matplotlib, scikit-image, and other libraries you might be interested in, and read
them regularly.

And when you do get stuck in your own work, don't be afraid to seek help there! We
are a friendly bunch! The *main* requirement when seeking help is to show that
you've tried a bit of problem solving yourself, and to provide others with a
minimal script and enough sample data to demonstrate your problem and how you've
tried to fix it.

- **No:** "I need to generate a big array of random Gaussians. Can someone
  help?"
- **No:** "I have this huge library at https://github.com/ron_obvious, if you
  look in the statistics library, there's a part that really needs random
  Gaussians, can someone take a look???"
- **Yes:** "I've been trying to generate a big list of random Gaussians like
  so: `gauss = [np.random.randn()] * 10**5`. But when I compute `np.mean(gauss)`,
  it's hardly ever as close to 0 as I expect. What am I doing wrong? The full
  script ready for copy-paste is below.

### GitHub

We also talked in the preface about GitHub. All of the code that we discussed lives
on GitHub:

- https://github.com/numpy/numpy
- https://github.com/scipy/scipy

and others. When something isn't working as you expect, it could be a bug. If,
after some investigation, you are convinced that you have indeed uncovered a
bug, you should go to the "issues" tab of the relevant GitHub repository and
create a new issue. This will ensure that the bug is on the radar of the
developers of that library, and that it will (hopefully) be fixed in the next
version. By the way, this advice also applies to "bugs" in the documentation:
if something in a library's documentation isn't clear to you, file an issue!

Even better than filing an issue is *submitting a pull request*. A pull request
improving a library's documentation is a great way to dip your toes in open
source! We can't cover the process here, but there are many books and resources
out there to help you:
- Anthony Scopatz and Katy Huff's [Effective Computation in
  Physics](http://shop.oreilly.com/product/0636920033424.do) covers git and
  GitHub, among many other topics in scientific computation.
- [Introducing GitHub](http://shop.oreilly.com/product/0636920033059.do) by
  Peter Bell and Brent Beer, covers GitHub in more detail.
- [Software Carpentry](https://software-carpentry.org/) has git lessons, and
  offers free workshops around the world throughout the year.
- Based partly on those lessons, one of your authors has created a complete
  tutorial on git and GitHub pull requests, [Open Source Science with Git and
  GitHub](http://jni.github.io/git-tutorial/).
- Finally, many open source projects on GitHub have a
  ["CONTRIBUTING"](https://github.com/scikit-image/scikit-image/blob/master/.github/CONTRIBUTING.txt)
  file, which contains a set of guidelines for contributing code to the project.

So, you are not starved for help on this topic!

We encourage you to contribute to the SciPy ecosystem as often as you can,
not only because you will help make these libraries better for all, but also
because it is one of the best ways to develop your coding abilities. With every
pull request you submit, you will get feedback about your code, helping you to
improve. You'll also become more familiar with the GitHub contributing process
and etiquette, which are highly valuable skills in today's job market.

### Conferences

In the same vein, we highly recommend attending a coding conference in this field.
SciPy, held every year in Austin, is fantastic, and probably your best bet if you
enjoyed this book. There's also a European version, EuroSciPy, which changes host
city every two years. Finally, the more general PyCon conference happens in the
US but also has offshoots around the world, such as PyCon-AU in Australia, which
has a "Science and Data" miniconference the day before the main conference.

Whichever conference you choose, *stay for the sprints* at the end of the
conference. A coding sprint is an intense session of team coding, and it is a
fantastic opportunity to learn the process of contributing to open source,
regardless of your skill level.  It is how one of your authors (Juan) started
in their open source journey.

## Beyond SciPy

The SciPy library is written not just in Python, but also in highly optimized C
and Fortran code that interfaces with Python. Together with NumPy, and related
libraries, it tries to cover most use cases that come up in scientific data
analysis, and provides very fast functions for these. Sometimes, however, a
scientific problem doesn't match at all with what's already in SciPy, and a pure
Python solution is too slow to be useful. What to do then?

[High-Performance Python](http://shop.oreilly.com/product/0636920028963.do),
by Micha Gorelick and Ian Ozsvald, covers what you need to know in these
situations: how to find where you *really* need performance, and the options
available to get that performance. We highly recommend it.

Here, we want to briefly mention two of those options that are particularly
relevant in the SciPy world.

First, Cython is a variant of Python that can be compiled to C, but then be
imported into Python. By providing some type annotations to Python variables,
the compiled C code can end up being a hundred or even a thousand times faster
than comparable Python code. Cython is now an industry standard and is used in
NumPy, SciPy, and many related libraries (such as scikit-image) to provide fast
algorithms in array-based code. Kurt Smith has written the simply-titled
[Cython](http://shop.oreilly.com/product/0636920033431.do) to teach you the
fundamentals of this language.

An often easier-to-use alternative to Cython is Numba, a just-in-time compiler
(JIT) for array-based Python. JITs wait for a function to be executed once, at
which point they can infer the types of all the function arguments and output,
and compile the code into a highly efficient form for those specific types. In
Numba code, you don't need to annotate types: Numba will infer them when a
function is first called. Instead, you simply need to make sure that you only
use basic types (integers, floats, etc), and arrays, rather than more
complicated Python objects. In these cases, Numba can compile the Python code
down to very efficient code and speed up computations by orders of magnitude.

Numba is still very young, but it is already very useful. Importantly, it shows
what is possible by Python JITs, which are set to become more commonplace: Python
3.6 added features to make it easier to use new JITs (the Pyjion JIT is based on
these). You can see some examples of Numba use, including how to combine it with
SciPy, in Juan's blog at https://ilovesymposia.com/tag/numba/. And Numba,
naturally, has its own very active and friendly mailing list.


## Contributing to this book

The source of this book is itself hosted on GitHub at
https://github.com/elegant-scipy/elegant-scipy (also at http://elegant-scipy.org).
Just as if you were contributing to any other open source project, you can raise
issues or submit pull requests if you find bugs or typos — and we would very much
appreciate it if you did!

We used some of the best code we could find to illustrate the various parts of
the SciPy and NumPy libraries. If you have a better example, please raise an issue
in the repo! We would love to include it in future editions.

We are also on Twitter, at [@elegantscipy](https://twitter.com/elegantscipy). Drop
us a line if you want to chat about the book! The individual authors are
[@jnuneziglesias](https://twitter.com/jnuneziglesias),
[@stefanvdwalt](https://twitter.com/stefanvdwalt), and
[@hdashnow](https://twitter.com/hdashnow).

We particularly want to hear about it if you use any of the ideas or code in this
book to advance your scientific research. That's the point of SciPy!

## Until next time...

In the meantime, we hope you enjoyed this book and found it useful. If so, tell
all your friends, and come say hi on the mailing lists, at a conference, on
GitHub, and on Twitter! Thanks for reading, and here's to even more Elegant SciPy!

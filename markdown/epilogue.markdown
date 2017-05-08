# Epilogue

> Quality means doing it right when no one is looking.
>
> — Henry Ford

Our main goal with this book was promoting elegant uses of the NumPy and SciPy
libraries, and, we hope, inspiring in you the feeling that quality code is
something worth striving for, while teaching you how to do effective scientific
analysis with SciPy.

## Where to next?

### Mailing lists

We mentioned in the preface that SciPy is a community. A great way to continue
learning is to subscribe to the main mailing lists for NumPy, SciPy, pandas,
matplotlib, scikit-image, and other libraries you might be interested in, and read
them regularly.

And when you do get stuck in your own work, don't be afraid to seek help there! We
are a friendly bunch! The *main* requirement when seeking help is to show that
you've tried a bit of problem solving yourself, and to provide others with a
minimal script to demonstrate your problem and how you've tried to fix it.

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

Even better than filing an issue is *submitting a pull request*. We can't cover
the process here, but there are many books and resources out there to help you:
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

What we would like to you is to encourage you to do this as often as you can,
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

In all these cases, *stay for the sprints* at the end of the conference. A coding
sprint is an intense session of team coding, and it is a fantastic opportunity to
learn the process of contributing to open source, regardless of your skill level.
It is how one of your authors (Juan) started in their open source journey.

## Contributing to this book

The source of this book is itself hosted on GitHub at
https://github.com/elegant-scipy/elegant-scipy (also at http://elegant-scipy.org).
Just as if you were contributing to any other open source project, you can raise
issues or submit pull requests if you find bugs or typos.

We used some of the best code we could find to illustrate the various parts of
the SciPy and NumPy libraries. If you have a better example, please raise an issue
in the repo! We would love to include it in future editions.

We are also on Twitter, at [@elegantscipy](https://twitter.com/elegantscipy). Drop
us a line if you want to chat about the book! The individual authors are
[@jnuneziglesias](https://twitter.com/jnuneziglesias),
[@stefanvdwalt](https://twitter.com/stefanvdwalt), and
[@hdashnow](https://twitter.com/hdashnow).

We particularly want to hear about it if you use any of the ideas or code in this
book to advance your scientific research! That's the point of SciPy!

In the meantime, we hope you enjoyed this book and found it useful. If so, tell
all your friends, and leave us a review! ;)

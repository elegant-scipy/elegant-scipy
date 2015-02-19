## Short version
We are [still looking](http://ilovesymposia.com/2015/02/04/call-for-code-nominations-for-elegant-scipy/) for code submissions meeting these criteria:
- Submissions **must** use NumPy, SciPy, or a closely related library in a non-trivial way.
- Submissions **must** be licensed as BSD, MIT, public domain, or something similarly liberal.
- Code should be satisfying in some way. ;) e.g. speed, conciseness, broad applicability...
- Preferably, nominate *someone else's* code that impressed *you*.
- Include a scientific application on real data.

Submit by one of:
- Twitter: mention [@hdashnow](https://twitter.com/hdashnow), [@stefanvdwalt](https://twitter.com/stefanvdwalt), or [@jnuneziglesias](https://twitter.com/jnuneziglesias), or just use the hashtag #ElegantSciPy;
- Email: [Stéfan van der Walt](mailto:stefanv(at)berkeley.edu), [Juan Nunez-Iglesias](mailto:juan.n@unimelb.edu.au), or [Harriet Dashnow](mailto:harriet.dashnow@unimelb.edu.au); or
- GitHub: create a new issue [here](https://github.com/HarrietInc/elegant-scipy-submissions/issues).

## Long version
A big thank you to everyone that has submitted code, retweeted, posted on mailing lists, etc! We're still pretty far from a book-length title though. I was also a bit vague about the kinds of submissions we wanted. I'll elaborate a bit on each of the above points:

### NumPy and SciPy use
Some excellent submissions did not use the SciPy library, but rather did amazing things with the Python standard library. I should have mentioned that the book will be specifically focused on the NumPy and SciPy libraries. That's just the scope of the book that O'Reilly has contracted us to write. Therefore, although we might try to fit examples of great broader Python uses into a chapter, they are not suitable to be the centerpieces.

We will make some exceptions, for example for very closely related libraries such as pandas and scikit-learn. But, generally, the scope is SciPy the library.

### Licensing
This one's pretty obvious. We can't use your submission if it's under a restrictive license.

### Submitting someone else's code
I suspect that a lot of people are shy about submitting their own code. Two things should alleviate this. First, you can now submit via email, so you don't have to be public about the self-promotion. (Not that there's anything wrong with that, but I know I sometimes struggle with it.) And second, I want to explicitly state that we *prefer it* if you submit *others'* code. This is not to discourage self-promotion, but to drive code quality even higher. It's a high bar to convince ourselves that our code is worthy of being called elegant, but it takes another level entirely to find someone else's code elegant! (The usual reaction when reading others' code is more like, "and what the *#$&^* is going on *here????*") So, try to think about times you saw great code during a code review, reading a blog post, or while grokking and fiddling with someone else's library.

### Data
Beautiful code is kind of a goal unto itself, but we really want to demonstrate how *useful* SciPy is in real scientific analysis. Therefore, although cute examples on synthetic data can illustrate quite well what a piece of code does, it would be extremely useful for us if you can point to examples with real scientific data behind them.

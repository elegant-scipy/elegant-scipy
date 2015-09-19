```python
%matplotlib inline
```
# Big data in little laptop: streaming data analysis with Python

Whenever I think too hard about streaming data analysis, my head hurts.

You have probably already done some streaming, perhaps without thinking about it these terms.
The simplest form is probably iterating through lines in a files, processing each line without ever reading the entire file into memory.
For example a loop like this:

```python
with open('data/expr.tsv') as f:
    for line in f:
        pass
```

This strategy works really well for cases where your problem can be neatly solved with by-row processing.
But things can quickly get out of hand when things get much more sophisticated.

In traditional programming models, you pass your data to a function, the function processes the data, and then returns the result.
Done.

But in streaming programs, a function processes *some* of the data, returns the processed chunk, then, while downstream functions are dealing with that chunk, the function receives a bit more, and so on...
All these things are going on at the same time!
How can one keep them straight?

For many years, I didn't.
But Matt Rocklin's blog posts on this topic really opened my eyes to the utility and elegance of streaming data analysis, to the point that it was impossible to contemplate writing this book without including a chapter on it.
Although streaming is not really a SciPy feature, it is a way of writing code that is critical to efficiently processing the large datasets that we see in science.
The Python language contains some very nice primitives for streaming data processing, and these can be combined with Matt's Toolz library to generate gorgeous, concise code that is extremely memory-efficient.
We will show you how to apply streaming concepts to make your SciPy code fast and and elegant.

Let me clarify what I mean by "streaming" and why you might want to do it.
Suppose you have some data in a CSV text file, and you want to compute the column-wise average of $\log(x+1)$ of the values.
The most common way to do this would be to use NumPy to load the values, compute the log function for all values in the full matrix, and then take the mean over the 1st axis:

```python
import numpy as np
expr = np.loadtxt('data/expr.tsv')
logexpr = np.log(expr + 1)
np.mean(logexpr, axis=1)
```

This works, and it follows a reassuringly familiar input-output model of computation.
But it's a pretty inefficient way to go about it!
We load the full matrix into memory (1), then make a copy with 1 added to each value (2), then make another copy to compute the log (3), before finally passing it on to `np.mean`.
That's three instances of the data array, to perform an operation that doesn't require keeping even *one* instance.
It's clear that for any kind of "big data" operation, this approach won't work.

Python's creators knew this, and created the "yield" keyword, which enables a function to process just one "sip" of the data, pass the result on to the next process, and *let the chain of processing complete* for that one piece of data before moving on to the next one.
"Yield" is a rather nice name for it: the function *yields* control to the next function, waiting to resume processing the data until all the downstream steps have processed that data point.

As I mentioned above, trying to think too hard about the flow of control in this paradigm is a surefire way to experience headaches, nausea, and other side effects.
An awesome feature of Python is that it abstracts this complexity away, allowing you to focus on the analysis functionality.
Here's how I think about it: for every processing function that would normally take a list (a collection of data) and transform that list, simply rewrite that function as taking a *stream* and *yielding* the result of every element of that stream.

Here's an example where we add 1 to each element in a list, using a standard data-copying method and a streaming method:

```python
def add1(elem):
    return elem + 1

def add1_all_standard(input):
    output = []
    for elem in input:
        output.append(add1(elem))
    return output

def add1_all_streaming(input_stream):
    for elem in input_stream:
        yield add1(elem)
```

```python
np.random.seed(seed=7)  # Set seed so we will get consistent results
```

```python
%%timeit
result = add1_all_standard(np.random.normal(0, 1, 1000))
```

```python
%%timeit
result = add1_all_streaming(np.random.normal(0, 1, 1000))
```

The advantage of the streaming approach is that elements of a stream aren't processed until they're needed, whether it's for computing a running sum, or for writing out to disk, or something else.
This can conserve a lot of memory when you have a lot of input items, or when each item is very big.
(Or both!)
This quote from one of Matt's posts very succinctly summarizes the utility of streaming data analysis:

> In my brief experience people rarely take this [streaming] route.
They use single-threaded in-memory Python until it breaks, and then seek out Big Data Infrastructure like Hadoop/Spark at relatively high productivity overhead.

Indeed, this describes my computational career perfectly, up until recent months.
But the intermediate approach can get you a *lot* farther than you think.
In some cases, it can get you there even faster than the supercomputing approach, by eliminating the overhead of multi-core communication and random-access to databases.
(For example, see [this post](http://www.frankmcsherry.org/graph/scalability/cost/2015/02/04/COST2.html) by Frank McSherry, where he processes a 128 billion edge graph on his laptop *faster* than using a graph database on a supercomputer.)

To clarify the flow of control when using streaming-style functions, it's useful to make *verbose* versions of the functions, which print out a message with each operation.

```python
import numpy as np

def tsv_line_to_array(line):
    lst = [float(elem) for elem in line.rstrip().split('\t')]
    return np.array(lst)

def readtsv_verbose(filename):
    print('starting readtsv')
    with open(filename) as fin:
        for i, line in enumerate(fin):
            print('reading line {}'.format(i))
            yield tsv_line_to_array(line)
    print('finished readtsv')

def add1_verbose(arrays_iter):
    print('starting adding 1')
    for i, arr in enumerate(arrays_iter):
        print('adding 1 to line {}'.format(i))
        yield arr + 1
    print('finished adding 1')

def log_verbose(arrays_iter):
    print('starting log')
    for i, arr in enumerate(arrays_iter):
        print('taking log of array {}'.format(i))
        yield np.log(arr)
    print('finished log')

def running_mean_verbose(arrays_iter):
    print('starting running mean')
    for i, arr in enumerate(arrays_iter):
        if i == 0:
            mean = arr
        mean += (arr - mean) / (i + 1)
        print('adding line {} to the running mean'.format(i))
    print('returning mean')
    return mean
```

Let's see it in action for a small sample file:

```python
fin = 'data/expr.tsv'
print('Creating lines iterator')
lines = readtsv_verbose(fin)
print('Creating loglines iterator')
loglines = log_verbose(add1_verbose(lines))
print('Computing mean')
mean = running_mean_verbose(loglines)
print('the mean log-row is: {}'.format(mean))
```

Note a few things:

- None of the computation is run when creating the lines and loglines iterators. This is because iterators are *lazy*, meaning they are not evaluated (or *consumed*) until a result is needed.
- When the computation is finally triggered, by the call to `running_mean_verbose`, it jumps back and forth between all the functions, as various computations are performed on each line, before moving on to the next line.

## Introducing the Toolz streaming library

This chapter's code example is from Matt Rocklin (who else?), in which he creates a Markov model from an entire human genome in 10 minutes on a laptop, using just a few lines of code.
(It has been slightly edited for easier downstream processing.)
Over the course of the chapter we'll actually augment it a little bit to start from compressed data (who wants to keep an uncompressed dataset on their hard drive?).
This modification is almost *trivial*, which speaks to the elegance of his example.

```python
import toolz as tz
from toolz import curried as c
from glob import glob
import itertools as it

LDICT = dict(zip('ACGTacgt', range(8)))
PDICT = {(a, b): (LDICT[a], LDICT[b])
         for a, b in it.product(LDICT, LDICT)}

def is_sequence(line):
    return not line.startswith('>')

def is_nucleotide(letter):
    return letter in LDICT  # ignore 'N'

@tz.curry
def increment_model(model, index):
    model[index] += 1


def genome(file_pattern):
    """Stream a genome, letter by letter, from a list of FASTA filenames."""
    return tz.pipe(file_pattern, glob, sorted,  # Filenames
                   c.map(open),  # lines
                   tz.concat,  # concatenate lines from all files
                   c.filter(is_sequence),  # drop header from each sequence
                   tz.concat,  # concatenate chars from all lines
                   c.filter(is_nucleotide))  # discard newlines and 'N'

def markov(seq):
    """Get a 1st-order Markov model from a sequence of nucleotides."""
    model = np.zeros((8, 8))
    tz.last(tz.pipe(seq,
                    c.sliding_window(2),  # each successive tuple
                    c.map(PDICT.__getitem__),  # location in matrix of tuple
                    c.map(increment_model(model))))  # increment matrix
    # convert counts to transition probability matrix
    model /= np.sum(model, axis=1)[:, np.newaxis]
    return model
```

We can then do the following to obtain a Markov model of repetitive sequences
in the fruit-fly genome:

```python
# dm6.fa.gz can be downloaded from ftp://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/
# Unzip before using: gzip -d dm6.fa.gz
dm = 'data/dm6.fa'
model = tz.pipe(dm, genome, c.take(1000000), markov)
```

There's a *lot* going on in that example, so we are going to unpack it little by little.
We'll actually run the example at the end of the chapter.

The first thing to note is how many functions come from the [Toolz library](http://toolz.readthedocs.org/en/latest/).
For example from Toolz we've used, `pipe`, `sliding_window`, `frequencies`, and a curried version of `map` (more on this later).
That's because Toolz is written specifically to take advantage of Python's iterators, and easily manipulate streams.

Let's start with `pipe`.
This function is simply syntactic sugar to make nested function calls easier to read.
This is important because that pattern becomes increasingly common when dealing with iterators.

As a simple example, let's rewrite our running mean using `pipe`:

```python
import toolz as tz
filename = 'data/expr.tsv'
mean = tz.pipe(filename, readtsv_verbose,
               add1_verbose, log_verbose, running_mean_verbose)

# This is equivalent to nesting the functions like this:
# running_mean_verbose(log_verbose(add1_verbose(readtsv_verbose(filename))))
```

What was originally multiple lines, or an unwieldy mess of parentheses, is now a clean description of the sequential transformations of the input data.
Much easier to understand!

This strategy also has an advantage over the original NumPy implementation: if we scale our data to millions or billions of rows, our computer might struggle to hold all the data in memory.
In contrast, here we are only loading lines from disk one at a time, and maintaining a single line's worth of data.

## k-mer counting and error correction

You might want to review chapters 1 and 2 for information about DNA and genomics.
Briefly, your genetic information, the blueprint for making *you*, is encoded as a sequence of chemical *bases* in your *genome*.
These are really, really tiny, so you can't just look in a microscope and read them.
You also can't read a long string of them: errors accumulate and the readout becomes unusable.
(New technology is changing this, but here we will focus on short-read sequencing data, the most common today.)
Luckily, every one of your cells has an identical copy of your genome, so what we can do is shred those copies into tiny segments (about 100 bases), and then assemble those like an enormous puzzle of 30 million pieces.

Before performing assembly, it is vital to perform read correction.
During DNA sequencing some bases are incorrectly read out, and must be fixed, or they will mess up the assembly.
(Imagine having puzzle pieces with the wrong shape.)

One correction strategy is to find similar reads in your dataset and fix the error by grabbing the correct information from those reads. Or alternatively, you may choose to completely discard those reads containing errors.

However, this is a very inefficient way to do this, because finding similar reads takes $N^2$ operations, or $9 \times 10^14$ for a 30 million read dataset!
(And these are not cheap operations.)

There is another way.
[REF: original k-mer/de-Bruijn Graph implementation] realized that reads could be broken down into smaller, overlapping *k-mers*, substrings of length k, which can then be stored in a hash table (a dictionary, in Python).
This has tons of advantages, but the main one is that instead of computing on the total number of reads, which can be arbitrarily large, we can compute on the total number of k-mers, which can only be as large as the genome itself — usually 1-2 orders of magnitude smaller than the reads.

If we choose a value for k that is large enough to ensure any k-mer appears only once in the genome, the number of times a k-mer appears is exactly the number of reads that originate from that part of the genome.
This is called the *coverage* of that region.

If a read has an error in it, there is a high probability that the k-mers overlapping the error will be unique or close to unique in the genome [REF].
Think of the equivalent in English: if you were to take reads from Shakespeare, and one read was "to be or nob to be", the 6-mer "nob to" will appear rarely or not at all, whereas "not to" will be very frequent.

This is the basis for k-mer error correction: split the reads into k-mers, count the occurrence of each k-mer, and use some logic to replace rare k-mers in reads with similar common ones.
(Or, alternatively, discard reads with erroneous k-mers.
This is possible because reads are so abundant that we can afford to toss out erroneous data.)

This is also an example in which streaming is *essential*.
As mentioned before, the number of reads can be enormous, so we don't want to store them in memory.

DNA sequence data is commonly represented in FASTA format.
This is a plaintext format, consisting of one or many DNA sequences per file, each with a name and the actual sequence.

A sample FASTA file:
    > sequence_name1
    ACGT

    > sequence_name2
    GACT

Now we have the required information to convert a stream of lines from a FASTA file to a count of k-mers:

- filter lines so that only sequence lines are used
- for each sequence line, produce a stream of k-mers
- add each k-mer to a dictionary counter

Here's how you would do this in pure Python, using nothing but built-ins:

```python
def is_sequence(line):
    line = line.rstrip()  # remove '\n' at end of line
    return len(line) > 0 and not line.startswith('>')

def reads_to_kmers(reads_iter, k=7):
     for read in reads_iter:
         for start in range(0, len(read) - k):
             yield read[start : start + k]  # note yeild, so this is a generator

def kmer_counter(kmer_iter):
    counts = {}
    for kmer in kmer_iter:
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts

with open('data/sample.fasta') as fin:
    reads = filter(is_sequence, fin)
    kmers = reads_to_kmers(reads)
    counts = kmer_counter(kmers)
```

This totally works and is streaming, so reads are loaded from disk one at a time and piped through the k-mer converter and to the k-mer counter.
We can then plot a histogram of the counts, and confirm that there are indeed two well-separated populations of correct and erroneous k-mers:

```python
from matplotlib import pyplot as plt

def integer_histogram(counts, normed=True, *args, **kwargs):
    hist = np.bincount(counts)
    if normed:
        hist = hist / np.sum(hist)
    return plt.bar(np.arange(len(hist)), hist, *args, **kwargs)

integer_histogram(list(counts.values()))
```

Notice the nice distribution of k-mer frequencies, along with a big bump of k-mers (at the left of the plot) that appear only once.
Such low frequency k-mers are likely to be errors.

But, with the code above, we are actually doing a bit too much work.
A lot of the functionality we wrote in for loops and yields is actually *stream manipulation*: transforming a stream of data into a different kind of data, and accumulating it at the end.
Toolz has a lot of stream manipulation primitives that make it easy to write the above in just one function call; and, once you know the names of the transforming functions, it also becomes easier to visualize what is happening to your data stream at each point.

For example, the *sliding window* function is exactly what we need to make k-mers:

```python
print(tz.sliding_window.__doc__)
```

Additionally, the *frequencies* function counts the appearance of individual items in a data stream!
Together with pipe, we can now count k-mers in a single function call (though we will still use our original FASTA parsing function):

```python
from toolz import curried as c

k = 7
counts = tz.pipe('data/sample.fasta', open, c.filter(is_sequence),
                 c.map(str.rstrip),
                 c.map(c.sliding_window(k)),
                 tz.concat, c.map(''.join),
                 tz.frequencies)
```

We neglected to discuss the *curried* part of this approach.

What does it mean to curry a function?
Normally in Python if you don't give a function all of its required arguments then it will throw a fit.
In contrast, a curried function can just take *some* of those arguments.
If the curried function doesn't get enough arguments, it returns a new function that takes the leftover arguments.
Once that second function is called with the remaining arguments it can perform the original task.
Another word for currying is partial evaluation.
We are evaluating part of the function.
In functional programming, currying is a way to produce a function that can wait for the rest of the arguments to show up later.

Currying is not named after the spice blend (though it does spice up your code).
It is named for Haskell Curry, the mathematician who invented the concept.
Haskell Curry is also the namesake of the Haskell programming language, which has functions curried by default!

Currying can be a bit of a mind-bend when you first start, so let's make our own curried function to see how it works.

```python
# First, let's write a simple function to curry
def my_sum(a, b):
    return a + b

my_sum(2, 5)
```

```python
# Now we write a curried version of my_sum
def my_sum_curried(a, b=None):

    def my_sum(a, b): # Here's our original function that needs to be curried
        return a + b

    if b is None:  # The second value is not given, so we will need to return a function

        def my_sum_partial(b):  # we're defining a function that takes a variable b,
            return my_sum(a, b)        # and uses the a variable that we already know about

        return my_sum_partial

    else:  # Both values were given, so we can just return a value
        return my_sum(a, b)
```

Now let's try out a curried function to make sure it does what we expect.

```python
my_sum_curried(2, 5)
```
Okay, it acts like a normal function when given both variables.
Now let's leave out the second variable.

```python
my_sum_curried(2)
```

It returned a function. Yay!
Now let's see if we can use that returned function as expected.

```python
partial_sum = my_sum_curried(2)
partial_sum(5)
```

Now that worked, but my_sum_curried was a reasonable hard function to read.
Future me will probably have trouble remembering how I wrote that code.
Luckily, Toolz has some syntactic sugar to help us out.

```python
import toolz as tz

@tz.curry          # We can use curry as a decorator
def curried_sum(x, y):
    return x + y

curried_sum_partial = curried_sum(2)    # curried_sum didn't receive enough arguments to evaluate
                                        # so it holds onto the 2 and waits, returning a
                                        # partially evaluated function
curried_sum_partial(5)
```

To summarize what we did, curried_sum is a curried function, so it can take one of the arguments and returns another function curried_sum_partial which “remembers” that argument.

In fact, all of the Toolz functions are also available as curried functions in the toolz.curried namespace.
Toolz also includes curried version of some handy higher order python functions like `map`, `filter` and `reduce`.
We will import the `curried` namespace as `c` so our code doesn't get too cluttered.
So for example the curried version of `map` will be `c.map`.
Note, that the curried functions (e.g. `c.map`) are different from the `@curry` decorator, which us used to create a curried function.

```python
from toolz import curried as c
c.map
```

As a reminder, `map` is a built-in function.
From the [docs](https://docs.python.org/3.4/library/functions.html#map):

> map(function, iterable, ...)
> Return an iterator that applies function to every item of iterable, yielding the results.

A curried version of `map` is particularly handy when working in a Toolz pipe.
You can just pass a function to `c.map` and then stream in the iterator later using `tz.pipe`.
Take another look at our function for reading in the genome to see how this works in practice.

```python
def genome(file_pattern):
    """Stream a genome from a list of FASTA filenames"""
    return tz.pipe(file_pattern, glob, sorted,        # Filenames
                                 c.map(open),         # Open each file
                                 c.map(tz.drop(1)),   # Drop header from each file
                                 concat,              # Concatenate all lines from all files together
                                 c.map(str.upper),    # Upper case each line
                                 c.map(str.strip),    # Strip off \n from each line
                                 concat)              # Concatenate all lines into one giant string sequence
```

Okay, so now we've got our heads around curried, let's get back to our k-mer counting code.
We can now observe the frequency of different k-mers:

```python
counts = np.fromiter(counts.values(), dtype=int, count=len(counts))
hist = np.bincount(counts)
plt.bar(np.arange(len(hist)), hist / hist.sum())
```



> ## tips {.callout}
>  - (list of list -> list) with tz.concat
>  - don’t get caught out:
>     * iterators get consumed.
So if you make a generator, do some processing, and then a later step fails, you need to re-create the generator.
The original is already gone.
>     * iterators are lazy; need to force evaluation sometimes.
>  - when you have lots of functions in a pipe, it’s sometimes hard to figure out where things go wrong.
Take a small stream and add functions to your pipe one by one from the first/leftmost until you find the broken one.

## Markov model from a full genome

- Intro to Markov models

In general, a Markov model assumes that the probability of the system moving to a given state, is only dependent on the state that it was in just previously.
For example if it is sunny right now, there is a high probability that it will be sunny tomorrow.
The fact that it was raining yesterday is irrelevant.
In this theory, all the information required to predict the future is encoded in the current state of things.
The past is irrelevant.
This assumption is useful for simplifying otherwise intractable problems.

- Where to download the human genome
- Matt's post and how to use it

## Image processing with streaming functions

[Image stream montage](https://github.com/microscopium/microscopium/blob/master/microscopium/preprocess.py#L848)

[Streaming illumination correction](https://github.com/microscopium/microscopium/blob/master/microscopium/preprocess.py#L587)

[Reservoir sampling](https://github.com/microscopium/microscopium/blob/master/microscopium/preprocess.py#L639)

If you take 1s/image, for 1M images you take about 12h.
Totally doable.
No need for cloud/compute clusters.

## Streaming PCA

sklearn has IncrementalPCA class.
But you need to chunk your data yourself.
Let’s make a function that can take a stream of data samples and perform PCA.
Be sure to look at the documentation for the class to understand some of the code below.

```python
import toolz as tz
from toolz import curried as c
from sklearn import decomposition
from sklearn import datasets
import numpy as np

def streaming_pca(samples, n_components=2, batch_size=100):
    ipca = decomposition.IncrementalPCA(n_components=n_components,
                                        batch_size=batch_size)
    # we use `tz.last` to force evaluation of the full iterator
    _ = tz.last(tz.pipe(samples,  # iterator of 1D arrays
                        c.partition(batch_size),  # iterator of tuples
                        c.map(np.array),  # iterator of 2D arrays
                        c.map(ipca.partial_fit)))  # partial_fit on each
    return ipca
```

```python
def array_from_txt(line, sep=',', dtype=np.float):
    return np.array(line.rstrip().split(sep), dtype=dtype)

with open('data/iris.csv') as fin:
    pca_obj = tz.pipe(fin, c.map(array_from_txt), streaming_pca)

with open('data/iris.csv') as fin:
    components = np.squeeze(list(tz.pipe(fin,
                                         c.map(array_from_txt),
                                         c.map(pca_obj.transform))))

from matplotlib import pyplot as plt
plt.scatter(*components.T)
```

# Getting the Markov model

Let's go back to the example and get that Markov model.

```
# We produced the model earlier like this:
# model = tz.pipe(dm, genome, markov)
print('    ', '      '.join('ACGTacgt'), '\n')
print(model)
```

We can also look at the result as an image:

```
plt.imshow(model, cmap='gist_heat', interpolation='nearest');
plt.colorbar();
ax = plt.gca()
ax.set_xticklabels(' ACGTacgt');
ax.set_yticklabels(' ACGTacgt');
```

Note how the G-A and G-C transitions are different between the repeat and
non-repeat parts of the genome. This information can be used to classify
previously unseen DNA sequence.

# Conclusions

- streaming in Python is easy when you use a few abstractions
- streaming can make you more productive:
	- big data takes linearly longer than small data (no nasty memory swapping)
    - don’t need a bigger machine
    - if your tests pass on small data, they’ll pass on big data
- streaming code is concise and readable using toolz (cytoolz for speed)
- Time to reiterate my take-home: think about whether you can stream over a dataset, and if you can, do it.
Your future self will thank you.
Doing it later is harder:

![To-do](https://pbs.twimg.com/media/CDxc6HTVIAAsiFO.jpg).

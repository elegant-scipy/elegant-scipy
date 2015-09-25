
## Code: in-memory with numpy
- Requirements: all met before start of chapter
- Rating: 5
- Notes: needs modification for actual data. Illustrates
  real-world row-by-row modification.

```python
import numpy as np
expr = np.loadtxt('expr.csv')
logexpr = log(expr + 1)
np.mean(logexpr, axis=1)
```

## Code: in-memory vs streaming with yield
- Requirements: yield
- Rating: 3
- Notes: No relationship to an actual analysis; using `process` is a bit
  strained.

```python
def process_full(input):
    output = []
    for elem in input:
        output.append(process(elem))
    return output

def process_streaming(input)
    for elem in input_stream:
        yield process(elem)
```

## Code: verbose streaming to display output
- Requirements: yield
- Rating: 5
- Notes: Few requirements and great for flow comprehension. Needs same dataset
  as "in-memory with numpy".

```python
import numpy as np

def csv_line_to_array(line):
    lst = [float(elem) for elem in line.rstrip()]
    return np.array(lst)

def loadtxt_verbose(filename):
    print('starting loadtxt')
    with open(filename) as fin:
        for i, line in enumerate(fin):
            print('reading line {}'.format(i))
            yield csv_line_to_array(line)
    print('finished loadtxt')

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

```python
# create small tempfile with small matrix of values
# and compute running mean
# lines = loadtxt_verbose(fin)
# loglines = log_verbose(add1_verbose(fin))
# mean = running_mean_verbose(loglines)
print('the mean log-row is: {}'.format(mean))
```

Code: log-mean using pipe
Requirements: iterators, toolz.pipe
Rating: 5
Notes: Natural introduction of pipe
```python
import toolz as tz
filename = 'foo.txt'
mean = tz.pipe(filename, open, read_csv_verbose,
               add1_verbose, log_verbose, running_mean_verbose)
```

## Code: streaming iterative PCA
- Requirements: PCA (previous chapters), sklearn (?), pipe, last, iterators are
  consumed, currying, partition, np.squeeze
- Rating: 5
- Notes: PCA is well-known and this IPCA implementation is pretty nice.
  However, it requires prior knowledge of many functions. Might need to be
  moved down. Question: how to introduce these functions one at a time?

```python
import toolz as tz
from toolz import curried
from sklearn import decomposition
from sklearn import datasets
import numpy as np

def streaming_pca(samples, n_components=2, batch_size=100):
    ipca = decomposition.IncrementalPCA(n_components=n_components,
                                        batch_size=batch_size)
    # we use `tz.last` to force evaluation of the full iterator
    _ = tz.last(tz.pipe(samples,  # iterator of 1D arrays
                        curried.partition(batch_size),  # iterator of tuples
                        curried.map(np.array),  # iterator of 2D arrays
                        curried.map(ipca.partial_fit)))  # partial_fit on each
    return ipca
```

```python
def array_from_txt(line, sep=',', dtype=np.float):
    return np.array(line.rstrip().split(sep), dtype=dtype)

with open('iris.csv') as fin:
    pca_obj = tz.pipe(fin, curried.map(array_from_txt), streaming_pca)

with open('iris.csv') as fin:
    components = np.squeeze(tz.pipe(fin,
                                    curried.map(array_from_txt),
                                    curried.map(pca_obj.transform)))

from matplotlib import pyplot as plt
plt.scatter(*components.T)
```

## Code: nucleotide transition probabilities in a few lines
- Requirements: genomes, Markov models, currying, drop, concat, str ops,
  sliding_window, glob, merge_with
- Rating: 5
- Notes: The line that inspired it all. It requires a lot of background and the
  payoff is just a 4x4 matrix of numbers... So it might need some sprucing-up

```python
from toolz import drop, pipe, sliding_window, merge_with, frequencies
from toolz.curried import map
from glob import glob

def genome(file_pattern):
    """Stream a genome from a list of FASTA filenames"""
    return pipe(file_pattern, glob, sorted,        # Filenames
                                 map(open),        # Open each file
                                 map(drop(1)),     # Drop header from each file
                                 concat,           # Concatenate all lines from all files together
                                 map(str.upper),   # Upper case each line
                                 map(str.strip),   # Strip off \n from each line
                                 concat)           # Concatenate all lines into one giant string sequence

def markov(seq):
    """Get a 2nd-order Markov model from a sequence"""
    return pipe(seq, sliding_window(3),          # Each successive triple{(A, A): {T: 10}}
                     frequencies,                # Count occurrences of each triple
                     dict.items, map(markov_reshape),   # Reshape counts so {(A, A, T): 10} -> {(A, A): {T: 10}}
                     merge_with(merge))          # Merge dicts from different pairs

def markov_reshape(item):
    ((a, b, c), count) = item
    return {(a, b): {c: count}}

if __name__ == '__main__':
    pipe('/home/mrocklin/data/human-genome/chr*.fa', genome, markov)
```

## Code: k-mer counting
- Requirements: genomics, reads, k-mers, numpy.bincount, filter, currying,
  sliding_window, concat, map, frequencies
- Rating: 4
- Notes: a fundamental operation in genomics, but a bit dry without downstream
  analysis, which is hard. Nice showcase for toolz functions.

```python
def is_sequence(line):
    line = line.rstrip()  # remove '\n' at end of line
    return len(line) > 0 and not line.startswith('>')

def reads_to_kmers(reads_iter, k=7):
     for read in reads_iter:
         for start in range(0, len(read) - k):
             yield read[start : start + k]

def kmer_counter(kmer_iter):
    counts = {}
    for kmer in kmer_iter:
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts

with open('reads.fasta') as fin:
    reads = filter(is_sequence, fin)
    kmers = reads_to_kmers(reads)
    counts = kmer_counter(kmer_iter)
```


```python
from matplotlib import pyplot as plt

def integer_histogram(counts, normed=True, *args, **kwargs):
    hist = np.bincount(counts)
    if normed:
        hist = hist / np.sum(hist)
    return plt.bar(np.arange(len(hist)), hist, *args, **kwargs)

integer_histogram(list(counts.values())
```

```python
print(tz.sliding_window.__doc__)
```

```python
from toolz import curried as cur

counts = tz.pipe('reads.fasta', open, cur.filter(is_sequence),
                 cur.map(cur.sliding_window(k)),
                 tz.concat, cur.map(''.join),
                 tz.frequencies)
```

## Code: 
Requirements: 
Rating: 
Notes: 

## Code: 
Requirements: 
Rating: 
Notes: 

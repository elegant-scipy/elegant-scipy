import numpy as np
from toolz import curried as tz

spliteach = tz.compose(tz.map, tz.curry(str.split))
tee = tz.compose(tz.map, tz.do)
arrayfromiter = tz.curry(np.fromiter)(dtype=float)

def getid_counts(lines, kind='symbol'):
    if kind == 'symbol':
        kind = 0
    else:
        kind = 1
    ids = []
    res = tz.pipe(lines, spliteach(sep='\t'), tz.pluck(0),
                         spliteach(sep='|'), tz.pluck(kind),
                         tee(ids.append), tz.frequencies)
    return ids, res


def sampleid_from_filename(filename):
    return '.'.join(filename.split('.')[:3])


def generate_counts_matrix(files):
    """Produce a counts matrix from a TCGA Level 3 archive.

    Parameters
    ----------
    files : list of string
        The input files to read from.

    Returns
    -------
    ids : list of string
        The row names (gene IDs).
    samples : list of string
        The column names (sample IDs).
    counts : array of float, shape (n_genes, n_samples)
        The expression counts.
    """
    ncols = len(files)
    with open(files[0]) as fin:
        header = next(fin)
        id_labels, id_counts = getid_counts(fin)
    # discard all non-unique ids
    nrows = len(id_labels)
    counts = np.zeros((nrows, ncols), dtype=np.float)
    for col, filename in enumerate(files):
        with open(filename) as fin:
            dat = tz.pipe(fin, tz.drop(1), spliteach(sep='\t'), tz.pluck(1),
                               tz.map(float), arrayfromiter(count=nrows))
            counts[:, col] = dat
    samples = list(map(sampleid_from_filename, files))
    rows_to_keep = [i for i, gene in enumerate(id_labels)
                      if id_counts[gene] == 1]
    ids = [id_labels[i] for i in rows_to_keep]
    counts = counts[rows_to_keep, :]
    return ids, samples, counts


if __name__ == '__main__':
    import sys
    import pandas as pd
    files = sys.argv[1:]
    ids, samples, counts = generate_counts_matrix(files)
    df = pd.DataFrame(data=counts, index=ids, columns=samples)
    df.to_csv('counts.txt', line_terminator='\n')

# Elegant NumPy: The Foundation of Scientific Python

> [NumPy] is everywhere. It is all around us. Even now, in this very room.
> You can see it when you look out your window or when you turn on your
> television. You can feel it when you go to work... when you go to church...
> when you pay your taxes.
>
> — Morpheus, *The Matrix*

This chapter touches on some statistical functions in [SciPy](http://www.scipy.org), but more than that, it focuses on exploring the NumPy array, a data structure that underlies almost all numerical scientific computation in Python.

```{code-block} python
def rpkm(counts, lengths):
    """Calculate reads per kilobase transcript per million reads.

    RPKM = (10^9 * C) / (N * L)

    Where:
    C = Number of reads mapped to a gene
    N = Total mapped reads in the experiment
    L = Exon length in base pairs for a gene

    Parameters
    ----------
    counts: array, shape (N_genes, N_samples)
        RNAseq (or similar) count data where columns are individual samples
        and rows are genes.
    lengths: array, shape (N_genes,)
        Gene lengths in base pairs in the same order
        as the rows in counts.

    Returns
    -------
    normed : array, shape (N_genes, N_samples)
        The RPKM normalized counts matrix.
    """
    # First, convert counts to float to avoid overflow when multiplying by
    # 1e9 in the RPKM formula
    C = counts.astype(float)
    N = np.sum(C, axis=0)  # sum each column to get total reads per sample
    L = lengths

    normed = 1e9 * C / (N[np.newaxis, :] * L[:, np.newaxis])

    return(normed)
```

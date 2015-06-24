#!/usr/bin/env python

"""genelen.py: get the length of "gene" features from GAF annotation file.

TCGA uses a specific gene annotation file for many of its projects, found at:
https://tcga-data.nci.nih.gov/tcgafiles/ftp_auth/distro_ftpusers/anonymous/other/GAF/GAF_bundle/outputs/TCGA.Sept2010.09202010.gaf

We needed to parse out the gene lengths from that file. This script extracts
gene symbol, gene ID, and gene length from it, only for gene features. (Not,
e.g., for SNPs.)
"""

from toolz import curried as tz
import pandas as pd


spliteach = tz.map(tz.curry(str.split)(sep='\t'))


def range2len(tup):
    _id, coords = tup
    symbol, _id = _id.split('|')[:2]
    length = int(coords.split('-')[-1])
    return (symbol, int(_id), length)


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    with open(filename) as fin:
        header = next(fin).rstrip().split('\t')
        geneid = header.index('FeatureID')
        genelen = header.index('FeatureCoordinates')
        feattype = header.index('FeatureType')
        output = tz.pipe(fin, spliteach,
                              tz.filter(lambda x: x[feattype] == 'gene'),
                              tz.pluck([geneid, genelen]),
                              tz.map(range2len), list)
    df = pd.DataFrame(output, columns=['GeneSymbol', 'GeneID', 'GeneLength'])
    df.to_csv('genes.csv')

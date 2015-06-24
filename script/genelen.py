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

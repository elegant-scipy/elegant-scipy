import sys
import os

sys.path.append(os.path.expanduser('~/projects/prin'))

from prin.parsers import arnetminer as arnet

filename_in = sys.argv[1]

g = arnet.parser(filename_in)

fnn = os.path.splitext(filename_in)[0] + '.nodes.txt'

with open(fnn, 'w') as fout:
    for n in g.nodes_iter():
        fout.write(g.node[n].get('description', n) + '\n')

fne = os.path.splitext(filename_in)[0] + '.csr.pickle'

import networkx as nx

gg = nx.to_scipy_sparse_matrix(g, weight=None, format='csr')

import pickle

with open(fne, 'wb') as fout:
    pickle.dump(gg, fout)

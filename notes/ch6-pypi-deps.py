import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt

chem = np.load('data/chem-network.npy')
gap = np.load('data/gap-network.npy')
neuron_ids = np.load('data/neurons.npy')
neuron_types = np.load('data/neuron-types.npy')
A = chem + gap
C = (A + A.T) / 2
n = C.shape[0]
D = np.zeros((n, n), dtype=np.float)
diag = (np.arange(n), np.arange(n))
D[diag] = np.sum(C, axis=0)
L = D - C
from scipy import linalg
b = np.sum(C * np.sign(A - A.T), axis=1)
z = linalg.pinv(L) @ b
Dinv2 = np.zeros((n, n))
Dinv2[diag] = D[diag] ** (-.5)
Q = Dinv2 @ L @ Dinv2
eigvals, eigvecs = linalg.eig(Q)
smallest_first = np.argsort(eigvals)
eigvals = eigvals[smallest_first]
eigvecs = eigvecs[:, smallest_first]
x = Dinv2 @ eigvecs[:, 1]
from matplotlib import colors

def plot_connectome(neuron_x, neuron_y, links, labels, types):
    colormap = colors.ListedColormap([[ 0.   ,  0.447,  0.698],
                                      [ 0.   ,  0.62 ,  0.451],
                                      [ 0.835,  0.369,  0.   ]])
    # plot neuron locations:
    points = plt.scatter(neuron_x, neuron_y, c=types, cmap=colormap,
                         edgecolors='face', zorder=1)

    # add text labels:
    for x, y, label in zip(neuron_x, neuron_y, labels):
        plt.text(x, y, '  ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=5, zorder=2)

    # plot links
    pre, post = np.nonzero(links)
    for src, dst in zip(pre, post):
        plt.plot(neuron_x[[src, dst]], neuron_y[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)

    plt.show()


y = Dinv2 @ eigvecs[:, 2]

import scipy.sparse.linalg
import networkx as nx
dependencies = nx.DiGraph()
with open('data/pypi-deps.txt') as lines:
    lib_dep_pairs = (str.split(line) for line in lines)
    dependencies.add_edges_from(lib_dep_pairs)
print('number of packages: ', dependencies.number_of_nodes())
print('number of dependencies: ', dependencies.number_of_edges())
print(max(dependencies.in_degree_iter(),
          key=lambda x: x[1]))
dependencies.remove_node('setuptools')

print(max(dependencies.in_degree_iter(),
          key=lambda x: x[1]))
packages_by_in = sorted(dependencies.in_degree_iter(),
                        key=lambda x: x[1], reverse=True)
print(packages_by_in[:40])
connected_packages = max(nx.connected_components(dependencies.to_undirected()),
                         key=len)
conn_dependencies = nx.subgraph(dependencies, connected_packages)
package_names = np.array(conn_dependencies.nodes())  # array for multi-indexing
adjacency_matrix = nx.to_scipy_sparse_matrix(conn_dependencies,
                                             dtype=np.float64)
n = len(package_names)
np.seterr(divide='ignore')  # ignore division-by-zero errors
from scipy import sparse

degrees = np.ravel(adjacency_matrix.sum(axis=1))
degrees_matrix = sparse.spdiags(1 / degrees, 0, n, n, format='csr')
transition_matrix = (degrees_matrix @ adjacency_matrix).T
from scipy.sparse.linalg.isolve import bicg  # biconjugate gradient solver
damping = 0.85
I = sparse.eye(n, format='csc')
pagerank, error = bicg(I - damping * transition_matrix,
                       (1-damping) / n * np.ones(n),
                       maxiter=int(1e4))
print('error code: ', error)
top = np.argsort(pagerank)[::-1]

print([package_names[i] for i in top[:40]])
def power(trans, damping=0.85, max_iter=int(1e5)):
    n = trans.shape[0]
    r0 = np.full(n, 1/n)
    r = r0
    for _ in range(max_iter):
        rnext = damping * trans @ r + (1 - damping) / n
        if np.allclose(rnext, r):
            print('converged')
            break
        r = rnext
    return r
def power2(trans, damping=0.85, max_iter=int(1e5)):
    n = trans.shape[0]
    is_dangling = np.ravel(trans.sum(axis=0) == 0)
    dangling = np.zeros(n)
    dangling[is_dangling] = 1 / n
    r0 = np.ones(n) / n
    r = r0
    for _ in range(max_iter):
        rnext = (damping * (trans @ r + dangling @ r) +
                 (1 - damping) / n)
        if np.allclose(rnext, r):
            return rnext
        else:
            r = rnext
    return r

def sub(mat, idxs):
    return mat[idxs, :][:, idxs]

n = 3000
top3k = top[:n]
names = package_names[top3k]
Adj = sub(adjacency_matrix, top3k)
Conn = (Adj + Adj.T) / 2

degrees = np.ravel(Conn.sum(axis=0))
Deg = sparse.spdiags(degrees, 0, n, n).tocsr()
Dinv2 = sparse.spdiags(degrees ** (-.5), 0, n, n).tocsr()
Lap = Deg - Conn
Affn = Dinv2 @ Lap @ Dinv2
eigvals, vec = sparse.linalg.eigsh(Affn, k=3, which='SM')
_, x, y = (Dinv2 @ vec).T

plt.figure()
plt.scatter(x, y)

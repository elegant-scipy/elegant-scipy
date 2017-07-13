import networkx as nx
import toolz as tz


@tz.curry
def add_edges(g_, edges_iter):
    for e in edges_iter:
        g_.add_edge(*e)
    return g_


def edge_from_kmer(kmer):
    return (kmer[:-1], kmer[1:])


def eulerian_path(g, algorithm='Fleury'):
    """Return a Eulerian path for the semi-eulerian graph ``g``.

    Parameters
    ----------
    g : nx.DiGraph, nx.MultiDiGraph
        The input graph. The graph must be semi-Eulerian.
    algorithm : {'Fleury', 'Hierholzer'}, optional
        Which algorithm to use to find the path. Hierholzer is faster
        but has the distinct disadvantage that I haven't implemented
        it yet. =P

    Returns
    -------
    edges_iter : iterator of edges
        An Eulerian path of ``g``.

    Notes
    -----
    An Eulerian path is one that visits every edge in a graph exactly
    once.
    """
    source, sink = _source_and_sink(g)
    g.add_edge(sink, source)
    edges_iter = nx.eulerian_circuit(g, source=source)
    return edges_iter


def _source_and_sink(g):
    """Find the source and sink of a semi-Eulerian graph

    Parameters
    ----------
    g : nx.DiGraph
        The input graph.

    Returns
    -------
    source, sink : nodes
        A node for which in-degree is 1 less than out-degree, and
        vice-versa.
    """
    source, sink = None, None
    for node in g.nodes_iter():
        indeg, outdeg = g.in_degree(node), g.out_degree(node)
        if indeg == outdeg - 1:
            source = node
            continue
        if outdeg == indeg - 1:
            sink = node
            continue
        if source is not None and sink is not None:
            break
    return source, sink

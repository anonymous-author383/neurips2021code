"""Module for the CsrGraph class"""
from emBench.graphs.csrDict import CsrDict
import numpy as np
import pickle as pkl


class CsrGraph(CsrDict):
    """Exposes an interface to deal with undirected, immutable graphs

    ATTRIBUTES
    ----------
    row_indxs : numpy.ndarray
        array containing the column ids for all the values stored
    row_ptrs : nump.ndarray
        array containing the pointers for new row locations into row_indxs

    METHODS
    -------
    degree(node)
        Returns the degree of node.
    eges()
        Returns an iterator over edges in graph.
    keys()
        Returns all nodes in the graph (same as nodes()).
    neighbors(v)
        Returns the neighbors of v.
    nodes()
        Returns an iterator over the nodes in the graph.
    number_of_edges()
        Returns the number of edges in the graph.
    number_of_nodes()
        Returns the number of nodes in the graph.
    remove_isolates()
        Returns a CsrGraph with all isolated nodes removed from the graph and
        relabels the remaining sequentially.
    save(filename)
        Write this object to python pickle file given by filename.
    to_csr_matrix()
        Returns adjacency matrix as scipy.sparse.csr_matrix.
    transpose()
        Returns the adjacency matrix as a CsrGraph.
    """


    def __init__(self, dictlike, **args):
        """
        PARAMETERS
        ----------
        dictlike : dict or dictlike
            a representation of the adjacency list keyed by node with neighbors
            as values
        """
        if 'symmetric' not in args:
            super().__init__(dictlike, True)
        else:
            if not args['symmetric']:
                raise ValueError('CsrGraph must be symmetric. Use CsrDict for\
                        directed graphs.')
            else:
                super().__init__(self, dictlike, **args)

    def number_of_nodes(self):
        """Returns the number of nodes

        RETURNS
        -------
        int"""
        return len(self)

    def nodes(self):
        """Returns iterator over nodes in graph (alias for self.keys())"""
        return self.keys()

    def neighbors(self, v):
        """Returns the neighbors of node v

        PARAMETERS
        ----------
        v: int
            index of node

        RETURNS
        -------
        numpy.ndarray containing indices of neighbors of v
        """
        return self[v]

    def degree(self, node):
        """Returns degree of node in graph

        PARAMETERS
        ----------
        node : int
            a node in the graph

        RETURNS
        -------
        int representing the size of node's neighborhood
        """
        return len(self[node])

    def number_of_edges(self):
        """Returns number of edges in graph"""
        return len(self.row_indxs) // 2

    def edges(self):
        """Returns iterator over all edges in graph"""
        edges = np.zeros([self.number_of_edges(), 2], dtype=np.int32)
        ind = 0
        for v in self.nodes():
            for u in self.neighbors(v)[self.neighbors(v) < v]:
                edges[ind, 0] = v
                edges[ind, 1] = u
                ind += 1
        return edges

    def remove_isolates(self):
        """Return value: SparseGraphWC that is isomorphic to self but has no zero degree nodes"""
        isolates = [v for v in self.nodes() if len(self.neighbors(v)) == 0]
        relabel_map = {v: v - len([y for y in isolates if y < v])
                       for v in self.nodes() if v not in isolates}
        adj = CsrDict({relabel_map[v]: [relabel_map[u] for u in self.neighbors(
            v)] for v in self.nodes() if v not in isolates})
        return CsrGraph(adj), relabel_map


    def save(self, filename):
        """save object to filename"""
        filep = open(filename, "wb")
        pkl.dump(self, filep)
        filep.close()

from emBench.graphs.labeledGraph import LabeledGraph
from emBench.graphs.csrDict import CsrDict
import numpy as np

class CommLabeledGraph(LabeledGraph):
    """Graphs with community labels over pairs.

    Pairs of nodes which share a community are labeled 1,
    while those that do not are labeled 0

    ATTRIBUTES
    ----------
    row_indxs : numpy.ndarray(dtype=np.int32, shape=(2m,))
        array containing the column ids for all the values stored
    row_ptrs : nump.ndarray(dtype=np.int32, shape=(n + 1,))
        array containing the pointers for new row locations into row_indxs

    METHODS
    -------
    keys()
        Returns all vertex ids
    to_csr_matrix()
        Returns a version of the CsrDict as a scipy.sparse.csr_matrix
    transpose()
        Returns a CsrDict keyed by values of self and with values equal
        to the set of keys in self which map to a given index.
    degree(node)
        Returns the degree of node.
    eges()
        Returns an iterator over edges in graph.
    label_pairs(pairs)
        Returns binary array of labels indicating whether
        the corresponding pairs are in the same community.
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
        write this object to python pickle file given by filename.
    """

    def __init__(self, adj, comm_indicator):
        """
        PARAMETERS
        ----------
        adj : any type that can be fed to CsrGraph
        comm_indicator: np.ndarray(dtype=np.bool_, shape=(n, c)
                Community indicator matrix where each row
                corresponds to a vertex and columns correspond to
                communities
        """
        super().__init__(adj)
        self._communities = comm_indicator

    def label_pairs(self, pairs, chunksize=4000):
        """Returns binary array of labels indicating whether
        the corresponding pairs are in the same community.

        PARAMETERS
        ----------
        pairs : np.ndarray(dtype=np.int32)
            array of shape (k, 2) containing pairs of nodes from graph
        chunksize=40000 : int
            parameter for controlling memory usage

        RETURNS
        -------
        np.ndarray(dtype=np.bool_, shape=(k,))
        """
        chunked_pairs = [pairs[i * chunksize:(i + 1) * chunksize]
                         for i in range(int(len(pairs) / chunksize))]
        if len(pairs) % chunksize != 0:
            chunked_pairs += [pairs[int(len(pairs) / chunksize) * chunksize:]]
        return np.hstack([self._label_pairs(batch) for batch in chunked_pairs])

    def _label_pairs(self, pairs):
        """edges: k x 2 numpy array of np.int32's representing pairs of vertices"""
        return np.hstack([np.dot(u, v.T)
                          for u, v in zip(self._communities[pairs[:, 0]],
                                          self._communities[pairs[:, 1]])])

"""Module for LabeledGraph abstract class"""
from abc import abstractmethod
from emBench.graphs.csrGraph import CsrGraph

class LabeledGraph(CsrGraph):
    """abstract class, can not be directly instantiated
    Extends CsrGraph to support edges_to_labels() method

    ATTRIBUTES
    ----------
    row_indxs : numpy.ndarray
        array containing the column ids for all the values stored
    row_ptrs : nump.ndarray
        array containing the pointers for new row locations into row_indxs

    METHODS
    -------
    keys()
        returns all keys in CsrDict
    to_csr_matrix()
        returns a version of the CsrDict as a scipy.sparse.csr_matrix
    transpose()
        returns a CsrDict keyed by values of self and with values equal
        to the set of keys in self which map to a given index
    degree(node)
        returns the degree of node
    eges()
        returns an iterator over edges in graph
    label_pairs(pairs)
        returns labels for pairs
    neighbors(v)
        returns the neighbors of v
    nodes()
        returns an iterator over the nodes in the graph
    number_of_edges()
        returns the number of edges in the graph
    number_of_nodes()
        returns the number of nodes in the graph
    remove_isolates()
        returns a CsrGraph with all isolated nodes removed from the graph and
        relabels the remaining sequentially
    save(filename)
        write this object to python pickle file given by filename
    """

    @abstractmethod
    def label_pairs(self, pairs, **kwargs):
        """get labels for pairs

        PARAMETERS
        ----------
        pairs : np.ndarray(dtype=np.int32)
            array of shape (k, 2) containing pairs of nodes from graph

        RETURNS
        -------
        np.ndarray(dtype=np.bool_, shape=(k,))
        """
        pass

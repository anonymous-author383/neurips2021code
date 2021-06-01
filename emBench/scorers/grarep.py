from emBench.scorers.embeddingScorerBase import _EmbeddingScorerBaseClass
import math
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from numba import njit, prange


@njit(parallel=True)
def _process_all_rows(data, cols, ptrs, new_data, new_cols, new_ptrs):
    n = len(ptrs) - 1
    new_data = np.log(data) - math.log(n)
    new_data[new_data > 0] = 0
    """
    for v in prange(n):
        new_data[ptrs[v]:ptrs[v + 1]] = np.log(data[ptrs[v]:ptrs[v + 1]])-math.log(n)
        new_data[new_data > 0] = 0
    """
    current_index = 0
    for v in range(n):
        new_ptrs[v] = current_index
        end_index = current_index + np.count_nonzero(new_data[ptrs[v]:ptrs[v + 1]])
        new_data[current_index:end_index] = new_data[ptrs[v]:ptrs[v + 1]][new_data[ptrs[v]:ptrs[v + 1]] < 0]
        new_data[end_index:ptrs[v + 1]] = 0
        new_cols[current_index:end_index] = cols[ptrs[v]:ptrs[v + 1]][new_data[ptrs[v]:ptrs[v + 1]] < 0]
        new_cols[end_index:ptrs[v + 1]] = 0
        current_index = end_index
    new_ptrs[-1] = current_index
    return new_data, new_cols, new_ptrs


class GraRepScorer(_EmbeddingScorerBaseClass):
    """An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Note: a lot of this is stolen from karateclub

    PARAMETERS
    ----------
        graph : CsrGraph
            graph to which we fit an embedding
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5.
    """
    def fit_to_graph(self, graph, dimensions=32, iteration=10, order=5, seed=42):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.seed = seed
        self.fit(graph)

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree(node) for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.csr_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **(A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacency matrices.
        """
        A = graph.to_csr_matrix()
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return (A_hat, A_hat)

    def _create_target_matrix(self):
        """
        Creating a log transformed target matrix.

        Return types:
            * **target_matrix** *(SciPy array)* - The PMI matrix.
        """
        self._A_tilde = self._A_tilde.dot(self._A_hat)
        """
        scores = np.log(self._A_tilde.data)-math.log(self._A_tilde.shape[0])
        rows = self._A_tilde.row[scores < 0]
        cols = self._A_tilde.col[scores < 0]
        scores = scores[scores < 0]
        """
        scores = np.zeros(len(self._A_tilde.data))
        new_cols = np.zeros(len(self._A_tilde.data), dtype=np.int32)
        new_indxs = np.zeros(len(self._A_hat.indptr), dtype=np.int32)
        scores, new_cols, new_indxs = _process_all_rows(self._A_tilde.data, self._A_tilde.indices, self._A_tilde.indptr, scores, new_cols, new_indxs)
        print(scores)
        print(new_cols)
        print(new_indxs)
        target_matrix = sparse.csr_matrix((scores, new_cols, new_indxs),
                                          dtype=np.float32)
        target_matrix.prune()
        return target_matrix

    def _create_single_embedding(self, target_matrix):
        """
        Fitting a single SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed)
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        self._embeddings.append(embedding)

    def fit(self, graph):
        """
        Fitting a GraRep model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._A_tilde, self._A_hat = self._create_base_matrix(graph)
        self._embeddings = []
        target_matrix = self._create_target_matrix()
        self._create_single_embedding(target_matrix)
        for step in range(self.order-1):
            target_matrix = self._create_target_matrix()
            self._create_single_embedding(target_matrix)
        self._emb = self.get_embedding()

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        embedding = np.concatenate(self._embeddings, axis=1)
        return embedding
